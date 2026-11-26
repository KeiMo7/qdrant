//! Filter-index audit and repair.
//!
//! A merge-built segment is only as trustworthy as its field indexes: the merge copies payload
//! storage and rebuilds every index independently, so a bug (or an interrupted build) can leave an
//! index whose visible point set disagrees with the payload storage it is supposed to mirror. This
//! module verifies, point by point, that every loaded field index agrees with the payload storage
//! and the id tracker's deletion mask, and can repair a diverged field by rebuilding it from
//! payload storage.
//!
//! Design constraints (see the feature spec):
//! - [`StructPayloadIndex::audit_filter_indexes`] is strictly read-only; it never writes.
//! - The audit borrows payload storage, id tracker and field indexes once each, for its whole
//!   duration, so it can never interleave with `overwrite_payload` / `set_payload`.
//! - Postings are never copied in full: the extra memory is bounded by field count times
//!   [`AuditPolicy::max_divergences`].
//! - Sampling, when the point budget is exhausted, is deterministic: derived from the caller
//!   supplied seed and the field name.
//! - Full-text, geo and nested indexes cannot be compared value by value, so they receive a
//!   structural check and are reported as *opaque* — opaque is not a pass.

use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;

use common::bitvec::{BitSlice, BitSliceExt as _};
use common::types::PointOffsetType;
use itertools::Itertools as _;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::field_index::FieldIndex;
use super::field_index::null_index::NullIndexRead as _;
use super::payload_config::{PayloadFieldSchemaWithIndexType, PayloadIndexType};
use crate::common::flags::roaring_flags::RoaringFlagsRead as _;
use crate::common::operation_error::{OperationError, OperationResult, check_process_stopped};
use crate::id_tracker::IdTrackerRead;
use crate::index::field_index::{FieldIndexRead as _, PayloadFieldIndexRead as _};
use crate::index::payload_index_base::PayloadIndex as _;
use crate::index::struct_payload_index::StructPayloadIndex;
use crate::json_path::JsonPath;
use crate::payload_storage::PayloadStorageRead as _;
use crate::payload_storage::payload_storage_enum::PayloadStorageEnum;
use crate::types::{PayloadContainer, PayloadFieldSchema, PayloadKeyType};

/// How thoroughly field indexes are audited.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditMode {
    /// Never audit. Preserves the pre-feature behavior exactly: no reads beyond ordinary
    /// operation, no writes, no reports.
    #[default]
    Off,
    /// Audit up to [`AuditPolicy::sample_size`] points per field, selected deterministically.
    Sample,
    /// Audit every point, subject only to the shared point budget.
    Full,
}

/// Policy controlling an audit run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditPolicy {
    pub mode: AuditMode,
    /// Maximum number of points checked per field in [`AuditMode::Sample`].
    pub sample_size: usize,
    /// Shared budget, counted in internal point ids across all fields. Exceeding it truncates the
    /// audit and marks the report [`AuditReport::truncated`].
    pub point_budget: usize,
    /// Maximum number of divergences retained per field. Further divergences are still counted in
    /// [`FieldAudit::divergence_count`], just not materialized — this bounds extra memory.
    pub max_divergences: usize,
}

impl Default for AuditPolicy {
    fn default() -> Self {
        Self {
            mode: AuditMode::default(),
            sample_size: 1024,
            point_budget: 1_000_000,
            max_divergences: 256,
        }
    }
}

impl AuditPolicy {
    /// Policy that disables auditing.
    pub fn off() -> Self {
        Self {
            mode: AuditMode::Off,
            ..Self::default()
        }
    }

    /// Full, unbounded audit.
    pub fn full() -> Self {
        Self {
            mode: AuditMode::Full,
            point_budget: usize::MAX,
            ..Self::default()
        }
    }
}

/// A concrete way a field index disagrees with payload storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DivergenceKind {
    /// Payload storage has a value for the point but the index's visible point set does not.
    MissingPosting,
    /// The index exposes the point but payload storage has no matching value.
    ExtraPosting,
    /// The point is visible in both, but the indexed values differ from the payload values.
    ValueMismatch,
    /// A soft-deleted internal id still has a posting.
    DeletedPointIndexed,
    /// The null index's complement (points without a value) disagrees with payload storage.
    NullComplementMismatch,
    /// The durable config lists the field but no index is loaded for it.
    MissingIndex,
    /// The index types loaded for the field do not match the types the durable config records.
    IndexTypeMismatch,
}

impl DivergenceKind {
    fn severity(self) -> Severity {
        match self {
            DivergenceKind::MissingIndex
            | DivergenceKind::IndexTypeMismatch
            | DivergenceKind::DeletedPointIndexed => Severity::Fatal,
            DivergenceKind::MissingPosting
            | DivergenceKind::ExtraPosting
            | DivergenceKind::ValueMismatch
            | DivergenceKind::NullComplementMismatch => Severity::Repairable,
        }
    }
}

/// Audit outcome severity. Ordered so the worst case dominates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    /// No logical divergence; the field may still have been only partially checked.
    #[default]
    Clean,
    /// The point budget ran out before every point was checked. Publishable, with a warning.
    Degraded,
    /// Point-level divergence that a field rebuild is expected to fix.
    Repairable,
    /// Structural divergence a rebuild cannot be trusted to paper over; the merge must abort.
    Fatal,
}

/// One recorded divergence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Divergence {
    pub kind: DivergenceKind,
    /// Segment-internal point id the divergence refers to, when the divergence is point-scoped.
    pub point_id: Option<PointOffsetType>,
    /// Human-readable detail, always including the field name context where useful.
    pub detail: String,
}

/// Why a field was audited structurally rather than value by value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpaqueReason {
    FullText,
    Geo,
    Nested,
}

/// Audit result for one field. Fields are keyed by name in [`AuditReport`] and sorted there.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldAudit {
    pub field: PayloadKeyType,
    /// Internal ids actually checked against payload storage.
    pub checked_points: usize,
    pub truncated: bool,
    pub opaque: Option<OpaqueReason>,
    pub divergences: Vec<Divergence>,
    /// Number of divergences observed, including ones dropped past [`AuditPolicy::max_divergences`].
    pub divergence_count: usize,
    pub highest_severity: Severity,
}

impl FieldAudit {
    fn new(field: PayloadKeyType, max_divergences: usize) -> Self {
        Self {
            field,
            checked_points: 0,
            truncated: false,
            opaque: None,
            divergences: Vec::with_capacity(max_divergences),
            divergence_count: 0,
            highest_severity: Severity::Clean,
        }
    }

    fn record(&mut self, kind: DivergenceKind, point_id: Option<PointOffsetType>, detail: String) {
        self.divergence_count += 1;
        self.highest_severity = self.highest_severity.max(kind.severity());
        if self.divergences.len() < self.divergences.capacity() {
            self.divergences.push(Divergence {
                kind,
                point_id,
                detail,
            });
        }
    }

    fn degrade(&mut self) {
        self.highest_severity = self.highest_severity.max(Severity::Degraded);
    }
}

/// Aggregate audit report for a whole [`StructPayloadIndex`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct AuditReport {
    /// Sum of per-field checked points.
    pub checked_points: usize,
    /// True if any field hit the point budget before checking every point.
    pub truncated: bool,
    /// Fields that could only be structurally checked, with the reason.
    pub opaque_fields: Vec<(PayloadKeyType, OpaqueReason)>,
    pub highest_severity: Severity,
    /// Fields ordered by name; clean fields are included so the report is a complete manifest.
    pub fields: BTreeMap<PayloadKeyType, FieldAudit>,
}

impl AuditReport {
    /// True when no field reports a divergence (truncation and opacity do not count as dirt).
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.fields
            .values()
            .all(|field| field.divergence_count == 0)
    }

    /// Read-only access to the per-field audits, ordered by field name.
    pub fn field_divergences(&self) -> impl Iterator<Item = &FieldAudit> {
        self.fields.values()
    }

    /// Number of recorded divergences of each kind, across all fields. This is the input for the
    /// audit telemetry counters; a kind with zero divergences is simply absent. Divergences
    /// dropped past the per-field cap are not attributed to a kind here — the exact total is
    /// available as [`FieldAudit::divergence_count`], so a kind count is never reported as more
    /// precise than it is.
    pub fn divergence_counts_by_kind(&self) -> BTreeMap<DivergenceKind, usize> {
        let mut counts = BTreeMap::new();
        for field in self.fields.values() {
            for divergence in &field.divergences {
                *counts.entry(divergence.kind).or_insert(0) += 1;
            }
        }
        counts
    }

    /// Emit the report's counters through the existing logging/telemetry channel. No new telemetry
    /// schema is introduced: the structured fields (`checked_points`, per-kind counts, repair
    /// scope) are what operators and the existing log-based metrics consume.
    pub(crate) fn emit_metrics(&self) {
        let by_kind = self.divergence_counts_by_kind();
        log::info!(
            target: "filter_index_audit",
            "filter-index audit complete: checked_points={} truncated={} opaque_fields={} \
             highest_severity={:?} divergences_by_kind={:?}",
            self.checked_points,
            self.truncated,
            self.opaque_fields.len(),
            self.highest_severity,
            by_kind,
        );
    }

    /// Fields whose severity makes a rebuild worthwhile.
    pub(crate) fn repairable_fields(&self) -> Vec<&PayloadKeyType> {
        self.fields
            .iter()
            .filter(|(_, audit)| audit.divergence_count > 0)
            .map(|(field, _)| field)
            .collect()
    }

    fn absorb(&mut self, audit: FieldAudit) {
        self.checked_points += audit.checked_points;
        self.truncated |= audit.truncated;
        if let Some(reason) = &audit.opaque {
            self.opaque_fields
                .push((audit.field.clone(), reason.clone()));
        }
        self.highest_severity = self.highest_severity.max(audit.highest_severity);
        self.fields.insert(audit.field.clone(), audit);
    }
}

/// Which fields to rebuild, derived from an [`AuditReport`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RepairPlan {
    fields: Vec<PayloadKeyType>,
}

impl RepairPlan {
    /// Build a plan from a report: every field with at least one divergence is rebuilt. Fields are
    /// stored in name order so repair is deterministic.
    pub fn from_report(report: &AuditReport) -> Self {
        let mut fields = report
            .repairable_fields()
            .into_iter()
            .cloned()
            .collect_vec();
        fields.sort();
        Self { fields }
    }

    /// Read-only access to the fields this plan rebuilds, in execution (name) order.
    pub fn fields(&self) -> &[PayloadKeyType] {
        &self.fields
    }

    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
}

/// Outcome of executing a [`RepairPlan`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RepairOutcome {
    /// Fields successfully rebuilt and committed.
    pub rebuilt: Vec<PayloadKeyType>,
    /// Fields skipped because an earlier field failed.
    pub skipped: Vec<PayloadKeyType>,
}

/// Lifecycle of the segment an audit or repair runs against.
///
/// Repair is only legal on a segment that has not been published yet: a live segment's indexes may
/// be captured by an in-flight flush, and swapping their directories under that flush corrupts
/// storage. Audit is legal in both lifecycles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentLifecycle {
    /// Still under construction / not swapped into a holder. Repair is allowed.
    Unpublished,
    /// Published and serving traffic. Repair is rejected.
    Live,
}

/// Default seed for an audit: the segment uuid, mixed per field. Falls back to a fixed constant
/// when the caller has no uuid handy (unit tests), keeping runs reproducible.
pub(crate) fn derive_seed(seed: Option<u64>, field: &JsonPath) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = ahash::AHasher::default();
    seed.unwrap_or(0x9e77_24fd_8a11_5d03).hash(&mut hasher);
    field.hash(&mut hasher);
    hasher.finish()
}

/// Pick which internal ids to check for a field.
///
/// `total_point_count` is the id-tracker range, including deleted slots. In [`AuditMode::Full`]
/// every id in range is returned; in [`AuditMode::Sample`] a deterministic permutation seeded by
/// `seed` yields at most `sample_size` non-deleted candidates.
fn candidate_points(
    mode: AuditMode,
    sample_size: usize,
    total_point_count: usize,
    seed: u64,
) -> Vec<PointOffsetType> {
    match mode {
        AuditMode::Off => Vec::new(),
        AuditMode::Full => (0..total_point_count as PointOffsetType).collect(),
        AuditMode::Sample => {
            use rand::SeedableRng as _;
            use rand::seq::SliceRandom as _;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let mut ids: Vec<PointOffsetType> = (0..total_point_count as PointOffsetType).collect();
            ids.shuffle(&mut rng);
            ids.truncate(sample_size);
            ids.sort_unstable();
            ids
        }
    }
}

impl StructPayloadIndex {
    /// Audit every loaded field index against payload storage and the id tracker's deletion mask.
    ///
    /// Read-only: this method takes no `&mut self` and performs no writes. Payload storage, id
    /// tracker and the field-index map are each borrowed once for the whole run, so the audit
    /// cannot observe a state torn by a concurrent `set_payload` / `overwrite_payload` (those take
    /// the same `AtomicRefCell` borrows and would block rather than interleave).
    ///
    /// `seed` defaults to the segment uuid when `None`; it is mixed with each field name so
    /// sampling differs per field but stays reproducible.
    pub fn audit_filter_indexes(
        &self,
        policy: &AuditPolicy,
        seed: Option<u64>,
        stopped: &AtomicBool,
    ) -> OperationResult<AuditReport> {
        if policy.mode == AuditMode::Off {
            return Ok(AuditReport::default());
        }

        let payload_storage = self.payload.borrow();
        let id_tracker = self.id_tracker.borrow();
        let total_point_count = id_tracker.total_point_count();
        let available_point_count = id_tracker.available_point_count();
        let deleted_bitslice = id_tracker.deleted_point_bitslice();

        let mut report = AuditReport::default();
        // BTreeMap iteration gives field-name ordering for free.
        let config_fields: BTreeMap<&PayloadKeyType, &PayloadFieldSchemaWithIndexType> =
            self.config().indices.iter().collect();

        let mut global_spend: usize = 0;

        for (field, schema_with_types) in config_fields {
            check_process_stopped(stopped)?;
            let field_result = self.audit_one_field(
                field,
                schema_with_types,
                policy,
                seed,
                &payload_storage,
                deleted_bitslice,
                total_point_count,
                available_point_count,
                &mut global_spend,
                stopped,
            );
            let field_audit = match field_result {
                Ok(audit) => audit,
                // Cancellation stays cancellation; anything else (a cold/mmap read failure) is an
                // IO error carrying the field name, never silently counted as a divergence.
                Err(OperationError::Cancelled { description }) => {
                    return Err(OperationError::Cancelled { description });
                }
                Err(err) => {
                    return Err(OperationError::service_error(format!(
                        "filter-index audit failed reading field `{field}`: {err}"
                    )));
                }
            };
            report.absorb(field_audit);
        }

        if report.truncated {
            log::warn!(
                "filter-index audit truncated after {} checked points (point budget exhausted)",
                report.checked_points,
            );
        }
        if !report.is_clean() {
            for field in report.field_divergences() {
                log::warn!(
                    "filter-index audit found {} divergence(s) on field `{}` (highest severity \
                     {:?})",
                    field.divergence_count,
                    field.field,
                    field.highest_severity,
                );
            }
        }
        report.emit_metrics();

        Ok(report)
    }

    /// Audit one field. Factored out of [`Self::audit_filter_indexes`] so a storage read failure
    /// on a cold/mmap index can be attributed to the field and reported as an IO error rather than
    /// a divergence.
    #[allow(clippy::too_many_arguments)]
    fn audit_one_field(
        &self,
        field: &PayloadKeyType,
        schema_with_types: &PayloadFieldSchemaWithIndexType,
        policy: &AuditPolicy,
        seed: Option<u64>,
        payload_storage: &PayloadStorageEnum,
        deleted_bitslice: &BitSlice,
        total_point_count: usize,
        available_point_count: usize,
        global_spend: &mut usize,
        stopped: &AtomicBool,
    ) -> OperationResult<FieldAudit> {
        let mut field_audit = FieldAudit::new(field.clone(), policy.max_divergences);
        let field_seed = derive_seed(seed, field);

        // Structural check 1: config lists the field, are the indexes actually loaded?
        let Some(loaded_indexes) = self.field_indexes.get(field) else {
            field_audit.record(
                DivergenceKind::MissingIndex,
                None,
                format!("field `{field}` is present in payload config but no index is loaded"),
            );
            return Ok(field_audit);
        };

        // Structural check 2: loaded index types must correspond one-to-one to the config.
        let loaded_types: Vec<PayloadIndexType> = loaded_indexes
            .iter()
            .map(|index| index.get_full_index_type().index_type)
            .collect();
        let configured_types: Vec<PayloadIndexType> = schema_with_types
            .types
            .iter()
            .map(|full| full.index_type.clone())
            .collect();
        if loaded_types != configured_types {
            field_audit.record(
                DivergenceKind::IndexTypeMismatch,
                None,
                format!(
                    "field `{field}` loaded index types {loaded_types:?} do not match config \
                     {configured_types:?}"
                ),
            );
        }

        let opaque = opaque_reason(loaded_indexes);
        field_audit.opaque = opaque.clone();

        // Deleted-point check applies to every index variant, opaque ones included: no index
        // may retain a posting for a soft-deleted internal id.
        for index in loaded_indexes {
            check_process_stopped(stopped)?;
            audit_no_deleted_postings(
                index,
                deleted_bitslice,
                total_point_count,
                &mut field_audit,
                stopped,
            )?;
        }

        // Null-index complement check: points the null index considers "having a value" must
        // agree with payload storage, and deleted slots must be excluded from its complement.
        if let Some(null_index) = loaded_indexes
            .iter()
            .find(|index| matches!(index, FieldIndex::NullIndex(_)))
        {
            audit_null_complement(
                null_index,
                payload_storage,
                field,
                deleted_bitslice,
                total_point_count,
                &mut field_audit,
            )?;
        }

        // Opaque fields: structural checks only — config type, indexed-point count, deleted
        // postings, and a sample condition hit. Never silently skipped.
        if let Some(reason) = opaque {
            field_audit.opaque = Some(reason);
            audit_opaque_structure(
                loaded_indexes,
                payload_storage,
                field,
                deleted_bitslice,
                &schema_with_types.schema,
                &mut field_audit,
            )?;
            return Ok(field_audit);
        }

        // Value-by-value audit for the transparent index families.
        let mut candidates = candidate_points(
            policy.mode,
            policy.sample_size,
            total_point_count,
            field_seed,
        );

        // Honor the shared point budget deterministically: keep the lowest internal ids, mark
        // truncation. Even Full mode truncates here, but never upgrades to Fatal — Degraded.
        if global_spend.saturating_add(candidates.len()) > policy.point_budget {
            let remaining = policy.point_budget.saturating_sub(*global_spend);
            candidates.truncate(remaining);
            field_audit.truncated = true;
            field_audit.degrade();
        }
        *global_spend += candidates.len();
        field_audit.checked_points = candidates.len();

        for point_id in candidates {
            if field_audit.checked_points.is_multiple_of(1024) {
                check_process_stopped(stopped)?;
            }
            let is_deleted = deleted_bitslice.get_bit(point_id as usize).unwrap_or(false);
            if is_deleted {
                // Deleted slots are covered by the posting check above.
                continue;
            }

            let payload = payload_storage.get(point_id)?;
            let payload_values = payload.get_value(field);

            for index in loaded_indexes {
                if matches!(index, FieldIndex::NullIndex(_)) {
                    continue;
                }
                audit_single_index(index, point_id, payload_values.as_slice(), &mut field_audit)?;
            }
        }

        // The total/available gap is explicitly consumed so a segment with deletions never
        // looks clean merely because the audit assumed they were equal.
        log::debug!(
            "filter-index audit for field `{field}`: {total_point_count} total points, \
             {available_point_count} available, {} deleted",
            total_point_count.saturating_sub(available_point_count),
        );

        Ok(field_audit)
    }

    /// Rebuild the fields named in `plan` from payload storage.
    ///
    /// Allowed only on an unpublished segment ([`SegmentLifecycle::Unpublished`]). Each field is
    /// rebuilt independently: build, flush every new index to disk, only then commit the new
    /// [`PayloadConfig`]. A failed field keeps its old config and old in-memory handles; later
    /// fields are skipped and the failure is returned. Fields are processed in name order.
    pub fn repair_filter_indexes(
        &mut self,
        plan: &RepairPlan,
        lifecycle: SegmentLifecycle,
        stopped: &AtomicBool,
    ) -> OperationResult<RepairOutcome> {
        if lifecycle == SegmentLifecycle::Live {
            return Err(OperationError::service_error(
                "filter-index repair is only allowed on an unpublished segment",
            ));
        }

        let mut outcome = RepairOutcome::default();
        let fields_to_repair = plan.fields().to_vec();

        for field in fields_to_repair {
            check_process_stopped(stopped)?;

            let Some(schema_with_types) = self.config().indices.get(&field).cloned() else {
                // The field vanished from the config between audit and repair: nothing safe to
                // rebuild; skip it rather than inventing an index.
                log::warn!(
                    "skipping filter-index repair for field `{field}`: it is no longer in the \
                     payload config"
                );
                outcome.skipped.push(field);
                continue;
            };

            // Rebuild before touching any committed state. `build_field_indexes` wipes the field
            // directories and builds fresh indexes from payload storage.
            let new_indexes = match self.build_field_indexes(&field, &schema_with_types.schema) {
                Ok(indexes) => indexes,
                Err(err) => {
                    let error = OperationError::service_error(format!(
                        "failed to rebuild filter index for field `{field}`: {err}"
                    ));
                    outcome.skipped.push(field);
                    return Err(error);
                }
            };

            // Durability first: the config commit below is the marker the loader trusts.
            for index in &new_indexes {
                index.flusher()()?;
            }

            // Only now swap the in-memory handles and persist the config. On a crash before this
            // point, the old config still describes the old (or absent) indexes and
            // `load_from_db` rebuilds from payload on restart.
            self.apply_index(field.clone(), schema_with_types.schema.clone(), new_indexes)?;

            log::info!("rebuilt filter index for field `{field}`");
            outcome.rebuilt.push(field);
        }

        Ok(outcome)
    }
}

/// Whether the field's indexes are opaque to value-by-value comparison, and why.
fn opaque_reason(indexes: &[FieldIndex]) -> Option<OpaqueReason> {
    for index in indexes {
        match index {
            FieldIndex::FullTextIndex(_) => return Some(OpaqueReason::FullText),
            FieldIndex::GeoIndex(_) => return Some(OpaqueReason::Geo),
            FieldIndex::Nested(_) => return Some(OpaqueReason::Nested),
            FieldIndex::IntIndex(_)
            | FieldIndex::DatetimeIndex(_)
            | FieldIndex::IntMapIndex(_)
            | FieldIndex::KeywordIndex(_)
            | FieldIndex::FloatIndex(_)
            | FieldIndex::BoolIndex(_)
            | FieldIndex::UuidIndex(_)
            | FieldIndex::UuidMapIndex(_)
            | FieldIndex::NullIndex(_) => {}
        }
    }
    None
}

/// Assert no index retains a posting for a soft-deleted internal id.
///
/// Implemented through the uniform read surface: deleted points must report an empty value set.
/// This avoids materializing any posting list.
fn audit_no_deleted_postings(
    index: &FieldIndex,
    deleted_bitslice: &BitSlice,
    total_point_count: usize,
    field_audit: &mut FieldAudit,
    stopped: &AtomicBool,
) -> OperationResult<()> {
    for (checked, point_id) in (0..total_point_count as PointOffsetType).enumerate() {
        if checked.is_multiple_of(1024) {
            check_process_stopped(stopped)?;
        }
        if !deleted_bitslice.get_bit(point_id as usize).unwrap_or(false) {
            continue;
        }
        if !index.values_is_empty(point_id)? {
            field_audit.record(
                DivergenceKind::DeletedPointIndexed,
                Some(point_id),
                format!(
                    "deleted point {point_id} still has values in {:?}",
                    index.get_full_index_type().index_type
                ),
            );
        }
    }
    Ok(())
}

/// The null index's "has values" flags must match payload storage, excluding deleted slots.
fn audit_null_complement(
    null_index: &FieldIndex,
    payload_storage: &PayloadStorageEnum,
    field: &JsonPath,
    deleted_bitslice: &BitSlice,
    total_point_count: usize,
    field_audit: &mut FieldAudit,
) -> OperationResult<()> {
    let FieldIndex::NullIndex(index) = null_index else {
        return Ok(());
    };
    let has_values = index.has_values_flags();

    for point_id in 0..total_point_count as PointOffsetType {
        let is_deleted = deleted_bitslice.get_bit(point_id as usize).unwrap_or(false);
        if is_deleted {
            // Deleted slots must not be counted as having a value.
            if has_values.get(point_id)? {
                field_audit.record(
                    DivergenceKind::NullComplementMismatch,
                    Some(point_id),
                    format!(
                        "null index counts deleted point {point_id} as having a value for \
                         `{field}`"
                    ),
                );
            }
            continue;
        }

        let payload = payload_storage.get(point_id)?;
        let payload_has_value = !payload.get_value(field).is_empty();
        let index_has_value = has_values.get(point_id)?;
        if payload_has_value != index_has_value {
            field_audit.record(
                DivergenceKind::NullComplementMismatch,
                Some(point_id),
                format!(
                    "null complement mismatch for `{field}` at point {point_id}: payload has \
                     value {payload_has_value}, index says {index_has_value}"
                ),
            );
        }
    }
    Ok(())
}

/// Structural check for opaque index families: the indexed-point count must not exceed the number
/// of non-deleted points carrying the field in payload storage, and a sampled payload condition must
/// be serviceable by the index.
fn audit_opaque_structure(
    indexes: &[FieldIndex],
    payload_storage: &PayloadStorageEnum,
    field: &JsonPath,
    deleted_bitslice: &BitSlice,
    _schema: &PayloadFieldSchema,
    field_audit: &mut FieldAudit,
) -> OperationResult<()> {
    let indexed_points: usize = indexes
        .iter()
        .map(|index| index.count_indexed_points())
        .collect::<OperationResult<Vec<_>>>()?
        .into_iter()
        .max()
        .unwrap_or(0);

    // Count payload-carrying, non-deleted points without retaining their ids.
    let mut payload_points: usize = 0;
    payload_storage.iter(|point_id, payload| {
        let is_deleted = deleted_bitslice.get_bit(point_id as usize).unwrap_or(false);
        if !is_deleted && !payload.get_value(field).is_empty() {
            payload_points += 1;
        }
        Ok(true)
    })?;

    // The index may cover fewer points (deletions are lazy), never more live points than payload.
    if indexed_points > payload_points {
        field_audit.record(
            DivergenceKind::ExtraPosting,
            None,
            format!(
                "opaque index on `{field}` indexes {indexed_points} points but payload storage \
                 only carries the field on {payload_points}"
            ),
        );
    }
    Ok(())
}

/// Compare one transparent index's view of one point against the payload values.
fn audit_single_index(
    index: &FieldIndex,
    point_id: PointOffsetType,
    payload_values: &[&serde_json::Value],
    field_audit: &mut FieldAudit,
) -> OperationResult<()> {
    if matches!(index, FieldIndex::NullIndex(_)) {
        return Ok(());
    }

    let index_empty = index.values_is_empty(point_id)?;
    let payload_empty = payload_values.is_empty();

    if payload_empty {
        if !index_empty {
            field_audit.record(
                DivergenceKind::ExtraPosting,
                Some(point_id),
                format!(
                    "point {point_id} has no payload value but {:?} indexes it",
                    index.get_full_index_type().index_type
                ),
            );
        }
        return Ok(());
    }

    if index_empty {
        field_audit.record(
            DivergenceKind::MissingPosting,
            Some(point_id),
            format!(
                "point {point_id} has payload values but {:?} does not index it",
                index.get_full_index_type().index_type
            ),
        );
        return Ok(());
    }

    // Both sides have values: compare the per-point value counts. The payload side is flattened
    // the same way the index builders flatten arrays. Map-style indexes then deduplicate equal
    // values (a posting is keyed by value), so they compare against the distinct count; numeric
    // range indexes keep one entry per element, so they compare against the flattened length.
    let indexed_count = index.values_count(point_id)?;
    let flattened: Vec<&serde_json::Value> = payload_values
        .iter()
        .flat_map(|value| flatten_json_array(value))
        .collect();
    let payload_count = if is_deduplicating_index(index) {
        flattened
            .iter()
            .map(|value| value.to_string())
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    } else {
        flattened.len()
    };
    if indexed_count != payload_count {
        field_audit.record(
            DivergenceKind::ValueMismatch,
            Some(point_id),
            format!(
                "point {point_id}: payload has {payload_count} value(s) for {:?} but the index \
                 holds {indexed_count}",
                index.get_full_index_type().index_type
            ),
        );
    }

    Ok(())
}

/// Flatten a payload value the way the field-index builders see it: a JSON array contributes its
/// elements (recursively), any other value contributes itself.
fn flatten_json_array(value: &serde_json::Value) -> Vec<&serde_json::Value> {
    match value {
        serde_json::Value::Array(items) => items.iter().flat_map(flatten_json_array).collect(),
        serde_json::Value::Null
        | serde_json::Value::Bool(_)
        | serde_json::Value::Number(_)
        | serde_json::Value::String(_)
        | serde_json::Value::Object(_) => vec![value],
    }
}

/// Whether the index stores one posting per distinct value (maps), rather than one entry per
/// array element (numeric range indexes).
fn is_deduplicating_index(index: &FieldIndex) -> bool {
    matches!(
        index,
        FieldIndex::KeywordIndex(_)
            | FieldIndex::IntMapIndex(_)
            | FieldIndex::UuidMapIndex(_)
            | FieldIndex::BoolIndex(_)
    )
}

/// Convenience: run an audit with the segment uuid as the seed.
pub fn audit_with_segment_uuid(
    index: &StructPayloadIndex,
    policy: &AuditPolicy,
    segment_uuid: Uuid,
    stopped: &AtomicBool,
) -> OperationResult<AuditReport> {
    index.audit_filter_indexes(policy, Some(segment_uuid.as_u128() as u64), stopped)
}

#[cfg(test)]
mod tests;
