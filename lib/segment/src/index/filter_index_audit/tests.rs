//! Unit tests for the filter-index audit and repair.

use std::str::FromStr;
use std::sync::atomic::AtomicBool;

use common::ambient;
use tempfile::Builder;

use super::*;
use crate::data_types::vectors::only_default_vector;
use crate::entry::entry_point::{
    NonAppendableSegmentEntry as _, SegmentEntry, StorageSegmentEntry as _,
};
use crate::index::field_index::FieldIndex;
use crate::index::filter_index_audit::{
    AuditMode, AuditPolicy, DivergenceKind, RepairPlan, SegmentLifecycle,
};
use crate::json_path::JsonPath;
use crate::segment_constructor::simple_segment_constructor::build_simple_segment;
use crate::types::{Distance, Payload, PayloadFieldSchema, PayloadSchemaType};

fn full_policy() -> AuditPolicy {
    AuditPolicy::full()
}

fn stopped() -> AtomicBool {
    AtomicBool::new(false)
}

/// Build an appendable segment with `n` points, every point carrying `payload`.
fn segment_with_payload(
    dir: &tempfile::TempDir,
    n: u32,
    payload: &str,
    field: &JsonPath,
    schema: PayloadFieldSchema,
) -> crate::segment::Segment {
    let _scope = ambient::test_guard();
    let mut segment = build_simple_segment(dir.path(), 4, Distance::Dot).unwrap();
    for i in 0..n {
        segment
            .upsert_point(
                u64::from(i) + 1,
                u64::from(i).into(),
                only_default_vector(&[1.0, 1.0, 1.0, 1.0]),
            )
            .unwrap();
        let payload: Payload =
            serde_json::from_str(&payload.replace("{id}", &i.to_string())).unwrap();
        segment
            .set_full_payload(u64::from(i) + 100, u64::from(i).into(), &payload)
            .unwrap();
    }
    segment
        .create_field_index(1000, field, Some(&schema))
        .unwrap();
    segment
}

/// Remove point 1's posting from every non-null index of `field`, simulating a lost posting.
fn drop_posting_for_point_one(segment: &crate::segment::Segment, field: &JsonPath) {
    let mut payload_index = segment.payload_index.borrow_mut();
    let indexes = payload_index.field_indexes.get_mut(field).unwrap();
    for index in indexes.iter_mut() {
        if matches!(index, FieldIndex::NullIndex(_)) {
            continue;
        }
        index.remove_point(1).unwrap();
    }
}

#[test]
fn clean_keyword_segment_audits_clean_and_reports_stable() {
    let _scope = ambient::test_guard();
    let dir = Builder::new().prefix("audit_kw").tempdir().unwrap();
    let field = JsonPath::from_str("color").unwrap();
    let segment = segment_with_payload(
        &dir,
        6,
        r#"{ "color": "value-{id}" }"#,
        &field,
        PayloadSchemaType::Keyword.into(),
    );

    let payload_index = segment.payload_index.borrow();
    let policy = full_policy();

    let first = payload_index
        .audit_filter_indexes(&policy, Some(42), &stopped())
        .unwrap();
    assert!(
        first.is_clean(),
        "a freshly built keyword index must audit clean: {first:?}"
    );
    assert_eq!(first.checked_points, 6);
    assert!(!first.truncated);
    assert!(first.opaque_fields.is_empty());

    // Repeated runs on the same input must produce identical reports.
    let second = payload_index
        .audit_filter_indexes(&policy, Some(42), &stopped())
        .unwrap();
    let third = payload_index
        .audit_filter_indexes(&policy, Some(42), &stopped())
        .unwrap();
    assert_eq!(first, second);
    assert_eq!(second, third);

    // Fields are reported ordered by name.
    let names: Vec<_> = first.fields.keys().cloned().collect();
    assert_eq!(names, vec![field.clone()]);
}

#[test]
fn missing_posting_on_keyword_is_detected_and_repaired() {
    let _scope = ambient::test_guard();
    let dir = Builder::new().prefix("audit_kw_bad").tempdir().unwrap();
    let field = JsonPath::from_str("color").unwrap();
    let segment = segment_with_payload(
        &dir,
        5,
        r#"{ "color": "value-{id}" }"#,
        &field,
        PayloadSchemaType::Keyword.into(),
    );

    drop_posting_for_point_one(&segment, &field);

    let report = {
        let payload_index = segment.payload_index.borrow();
        payload_index
            .audit_filter_indexes(&full_policy(), Some(7), &stopped())
            .unwrap()
    };
    assert!(!report.is_clean());
    let field_audit = report.fields.get(&field).unwrap();
    assert!(
        field_audit
            .divergences
            .iter()
            .any(|d| d.kind == DivergenceKind::MissingPosting && d.point_id == Some(1))
    );
    // Field-level ordering: divergences keep the point id they refer to.
    let point_ids: Vec<_> = field_audit
        .divergences
        .iter()
        .filter_map(|d| d.point_id)
        .collect();
    let mut sorted = point_ids.clone();
    sorted.sort_unstable();
    assert_eq!(point_ids, sorted);

    // Repair the unpublished segment and re-audit.
    let plan = RepairPlan::from_report(&report);
    assert_eq!(plan.fields(), vec![field.clone()]);
    {
        let mut payload_index = segment.payload_index.borrow_mut();
        let outcome = payload_index
            .repair_filter_indexes(&plan, SegmentLifecycle::Unpublished, &stopped())
            .unwrap();
        assert_eq!(outcome.rebuilt, vec![field.clone()]);
        assert!(outcome.skipped.is_empty());
    }
    let clean = segment
        .payload_index
        .borrow()
        .audit_filter_indexes(&full_policy(), Some(7), &stopped())
        .unwrap();
    assert!(
        clean.is_clean(),
        "repaired keyword index must audit clean: {clean:?}"
    );
}

#[test]
fn repair_is_rejected_on_a_live_segment() {
    let _scope = ambient::test_guard();
    let dir = Builder::new().prefix("audit_live").tempdir().unwrap();
    let field = JsonPath::from_str("n").unwrap();
    let segment = segment_with_payload(
        &dir,
        3,
        r#"{ "n": {id} }"#,
        &field,
        PayloadSchemaType::Integer.into(),
    );
    drop_posting_for_point_one(&segment, &field);

    let report = segment
        .payload_index
        .borrow()
        .audit_filter_indexes(&full_policy(), Some(1), &stopped())
        .unwrap();
    let plan = RepairPlan::from_report(&report);

    let err = segment
        .payload_index
        .borrow_mut()
        .repair_filter_indexes(&plan, SegmentLifecycle::Live, &stopped())
        .unwrap_err();
    assert!(
        err.to_string().contains("unpublished segment"),
        "live repair must be refused, got: {err}"
    );
}

#[test]
fn integer_bool_uuid_datetime_fields_audit() {
    let dir = Builder::new().prefix("audit_typed").tempdir().unwrap();
    let _scope = ambient::test_guard();
    let mut segment = build_simple_segment(dir.path(), 4, Distance::Dot).unwrap();

    let cases: Vec<(JsonPath, PayloadSchemaType, String)> = vec![
        (
            "count".parse().unwrap(),
            PayloadSchemaType::Integer,
            r#"{"count": {id}}"#.into(),
        ),
        (
            "flag".parse().unwrap(),
            PayloadSchemaType::Bool,
            r#"{"flag": true}"#.into(),
        ),
        (
            "uid".parse().unwrap(),
            PayloadSchemaType::Uuid,
            r#"{"uid": "550e8400-e29b-41d4-a716-44665544000{id}"}"#.into(),
        ),
        (
            "at".parse().unwrap(),
            PayloadSchemaType::Datetime,
            r#"{"at": "2024-01-{day}T12:00:00Z"}"#.into(),
        ),
    ];

    for i in 0..4u32 {
        segment
            .upsert_point(
                u64::from(i) + 1,
                u64::from(i).into(),
                only_default_vector(&[0.; 4]),
            )
            .unwrap();
        let mut payload = Payload::default();
        for (_, _, template) in &cases {
            let part: Payload = serde_json::from_str(
                &template
                    .replace("{id}", &i.to_string())
                    .replace("{day}", &(i + 1).to_string()),
            )
            .unwrap();
            payload.0.extend(part.0);
        }
        segment
            .set_full_payload(u64::from(i) + 10, u64::from(i).into(), &payload)
            .unwrap();
    }
    for (field, schema_type, _) in &cases {
        segment
            .create_field_index(200, field, Some(&PayloadFieldSchema::from(*schema_type)))
            .unwrap();
    }

    let report = segment
        .payload_index
        .borrow()
        .audit_filter_indexes(&full_policy(), Some(99), &stopped())
        .unwrap();
    assert!(
        report.is_clean(),
        "all typed indexes must audit clean: {report:?}"
    );
    let mut names: Vec<_> = report.fields.keys().map(|k| k.to_string()).collect();
    names.sort();
    assert_eq!(names, vec!["at", "count", "flag", "uid"]);

    // Corrupt the integer index and confirm the precise divergence is reported.
    {
        let mut index = segment.payload_index.borrow_mut();
        let indexes = index.field_indexes.get_mut(&cases[0].0).unwrap();
        for field_index in indexes.iter_mut() {
            if matches!(field_index, FieldIndex::NullIndex(_)) {
                continue;
            }
            field_index.remove_point(2).unwrap();
        }
    }
    let bad = segment
        .payload_index
        .borrow()
        .audit_filter_indexes(&full_policy(), Some(99), &stopped())
        .unwrap();
    let integer_audit = bad.fields.get(&cases[0].0).unwrap();
    assert!(
        integer_audit
            .divergences
            .iter()
            .any(|d| d.kind == DivergenceKind::MissingPosting && d.point_id == Some(2))
    );
}

#[test]
fn geo_full_text_and_nested_fields_are_opaque_not_silently_skipped() {
    let dir = Builder::new().prefix("audit_opaque").tempdir().unwrap();
    let _scope = ambient::test_guard();
    let mut segment = build_simple_segment(dir.path(), 4, Distance::Dot).unwrap();

    for i in 0..4u32 {
        segment
            .upsert_point(
                u64::from(i) + 1,
                u64::from(i).into(),
                only_default_vector(&[0.; 4]),
            )
            .unwrap();
        let payload: Payload = serde_json::from_str(&format!(
            r#"{{
                "location": {{ "lat": 52.0, "lon": 13.{i} }},
                "description": "the quick brown fox {i}",
                "items": [{{ "price": {i} }}, {{ "price": {i} }}]
            }}"#
        ))
        .unwrap();
        segment
            .set_full_payload(u64::from(i) + 10, u64::from(i).into(), &payload)
            .unwrap();
    }

    let geo: JsonPath = "location".parse().unwrap();
    let text: JsonPath = "description".parse().unwrap();
    let nested: JsonPath = "items[].price".parse().unwrap();

    segment
        .create_field_index(100, &geo, Some(&PayloadSchemaType::Geo.into()))
        .unwrap();
    segment
        .create_field_index(101, &text, Some(&PayloadSchemaType::Text.into()))
        .unwrap();
    // A per-element (nested) index: the schema names the array key explicitly.
    let nested_schema: PayloadFieldSchema = PayloadSchemaType::Integer.into();
    let nested_schema = nested_schema.nested_in("items");
    segment
        .create_field_index(102, &nested, Some(&nested_schema))
        .unwrap();

    let report = segment
        .payload_index
        .borrow()
        .audit_filter_indexes(&full_policy(), Some(5), &stopped())
        .unwrap();

    let opaque_names: Vec<_> = report
        .opaque_fields
        .iter()
        .map(|(field, _)| field.to_string())
        .collect();
    assert!(opaque_names.contains(&"location".to_string()));
    assert!(opaque_names.contains(&"description".to_string()));
    assert!(opaque_names.contains(&"items[].price".to_string()));
    // Opaque fields still get a full field audit with structural checks.
    for name in ["location", "description", "items[].price"] {
        let key: JsonPath = name.parse().unwrap();
        assert!(
            report.fields.contains_key(&key),
            "opaque field {name} must still be reported"
        );
    }
    assert!(
        report.is_clean(),
        "structural checks of fresh indexes must pass: {report:?}"
    );
}

#[test]
fn deleted_points_are_excluded_from_every_index() {
    let _scope = ambient::test_guard();
    let dir = Builder::new().prefix("audit_deleted").tempdir().unwrap();
    let field: JsonPath = "color".parse().unwrap();
    let mut segment = segment_with_payload(
        &dir,
        4,
        r#"{ "color": "c-{id}" }"#,
        &field,
        PayloadSchemaType::Keyword.into(),
    );

    // Soft-delete point 2.
    segment.delete_point(500, (2u64).into()).unwrap();

    let report = segment
        .payload_index
        .borrow()
        .audit_filter_indexes(&full_policy(), Some(11), &stopped())
        .unwrap();
    assert!(
        report.is_clean(),
        "a properly maintained index must not retain the deleted point: {report:?}"
    );

    // Now corrupt: re-add a posting for the deleted point directly on the map index.
    {
        let mut index = segment.payload_index.borrow_mut();
        let indexes = index.field_indexes.get_mut(&field).unwrap();
        for field_index in indexes.iter_mut() {
            if matches!(field_index, FieldIndex::NullIndex(_)) {
                continue;
            }
            let value = [serde_json::Value::String("c-2".to_string())];
            let refs: Vec<&serde_json::Value> = value.iter().collect();
            field_index.add_point(2, &refs).unwrap();
        }
    }
    let bad = segment
        .payload_index
        .borrow()
        .audit_filter_indexes(&full_policy(), Some(11), &stopped())
        .unwrap();
    let audit = bad.fields.get(&field).unwrap();
    assert!(
        audit
            .divergences
            .iter()
            .any(|d| d.kind == DivergenceKind::DeletedPointIndexed && d.point_id == Some(2))
    );
    assert_eq!(audit.highest_severity, Severity::Fatal);
}

#[test]
fn empty_segment_and_zero_index_segment_report_clean() {
    let dir = Builder::new().prefix("audit_empty").tempdir().unwrap();
    let _scope = ambient::test_guard();
    let segment = build_simple_segment(dir.path(), 4, Distance::Dot).unwrap();

    let report = segment
        .payload_index
        .borrow()
        .audit_filter_indexes(&full_policy(), Some(1), &stopped())
        .unwrap();
    assert!(report.is_clean());
    assert_eq!(report.checked_points, 0);
}

#[test]
fn sample_mode_respects_budget_and_marks_truncated() {
    let _scope = ambient::test_guard();
    let dir = Builder::new().prefix("audit_budget").tempdir().unwrap();
    let field: JsonPath = "n".parse().unwrap();
    let segment = segment_with_payload(
        &dir,
        10,
        r#"{ "n": {id} }"#,
        &field,
        PayloadSchemaType::Integer.into(),
    );

    let policy = AuditPolicy {
        mode: AuditMode::Sample,
        sample_size: 3,
        point_budget: 1000,
        max_divergences: 8,
    };
    let report = segment
        .payload_index
        .borrow()
        .audit_filter_indexes(&policy, Some(123), &stopped())
        .unwrap();
    assert_eq!(report.checked_points, 3);
    assert!(!report.truncated);

    // A budget of one point forces truncation.
    let tight = AuditPolicy {
        mode: AuditMode::Full,
        sample_size: 3,
        point_budget: 1,
        max_divergences: 8,
    };
    let truncated = segment
        .payload_index
        .borrow()
        .audit_filter_indexes(&tight, Some(123), &stopped())
        .unwrap();
    assert!(truncated.truncated);
    assert_eq!(truncated.checked_points, 1);
    assert_eq!(truncated.highest_severity, Severity::Degraded);
    // Truncation alone never escalates to Fatal and does not dirty the report.
    assert!(truncated.is_clean());
}

#[test]
fn off_mode_is_a_noop() {
    let _scope = ambient::test_guard();
    let dir = Builder::new().prefix("audit_off").tempdir().unwrap();
    let field: JsonPath = "n".parse().unwrap();
    let segment = segment_with_payload(
        &dir,
        3,
        r#"{ "n": {id} }"#,
        &field,
        PayloadSchemaType::Integer.into(),
    );
    drop_posting_for_point_one(&segment, &field);

    let report = segment
        .payload_index
        .borrow()
        .audit_filter_indexes(&AuditPolicy::off(), Some(1), &stopped())
        .unwrap();
    assert!(report.is_clean());
    assert_eq!(report.checked_points, 0);
}

#[test]
fn cancelled_audit_returns_cancelled_and_no_partial_report() {
    let _scope = ambient::test_guard();
    let dir = Builder::new().prefix("audit_cancel").tempdir().unwrap();
    let field: JsonPath = "n".parse().unwrap();
    let segment = segment_with_payload(
        &dir,
        4,
        r#"{ "n": {id} }"#,
        &field,
        PayloadSchemaType::Integer.into(),
    );
    let stopped = AtomicBool::new(true);
    let err = segment
        .payload_index
        .borrow()
        .audit_filter_indexes(&full_policy(), Some(1), &stopped)
        .unwrap_err();
    assert!(matches!(
        err,
        crate::common::operation_error::OperationError::Cancelled { .. }
    ));
}

#[test]
fn max_divergences_bounds_materialized_divergences() {
    let _scope = ambient::test_guard();
    let dir = Builder::new().prefix("audit_cap").tempdir().unwrap();
    let field: JsonPath = "n".parse().unwrap();
    let segment = segment_with_payload(
        &dir,
        8,
        r#"{ "n": {id} }"#,
        &field,
        PayloadSchemaType::Integer.into(),
    );
    // Lose every posting except point 0's: seven divergences, but the report retains only two.
    {
        let mut index = segment.payload_index.borrow_mut();
        let indexes = index.field_indexes.get_mut(&field).unwrap();
        for field_index in indexes.iter_mut() {
            if matches!(field_index, FieldIndex::NullIndex(_)) {
                continue;
            }
            for point_id in 1..8u32 {
                field_index.remove_point(point_id).unwrap();
            }
        }
    }
    let policy = AuditPolicy {
        mode: AuditMode::Full,
        sample_size: 0,
        point_budget: usize::MAX,
        max_divergences: 2,
    };
    let report = segment
        .payload_index
        .borrow()
        .audit_filter_indexes(&policy, Some(1), &stopped())
        .unwrap();
    let audit = report.fields.get(&field).unwrap();
    // Seven points lost, counted once per non-null index on the field (so a multiple of seven).
    assert_eq!(audit.divergence_count % 7, 0);
    assert!(audit.divergence_count >= 7);
    // Materialized divergences are capped by max_divergences regardless of the true count.
    assert_eq!(audit.divergences.len(), 2);
}

#[test]
fn null_complement_mismatch_is_detected() {
    let _scope = ambient::test_guard();
    let dir = Builder::new().prefix("audit_null").tempdir().unwrap();
    let field: JsonPath = "color".parse().unwrap();
    let segment = segment_with_payload(
        &dir,
        3,
        r#"{ "color": "c-{id}" }"#,
        &field,
        PayloadSchemaType::Keyword.into(),
    );

    // Corrupt the null index directly: mark a value-carrying point as null-only by dropping it
    // from the null index's has-values flags.
    {
        let mut index = segment.payload_index.borrow_mut();
        let indexes = index.field_indexes.get_mut(&field).unwrap();
        for field_index in indexes.iter_mut() {
            if let FieldIndex::NullIndex(null) = field_index {
                null.remove_point(0).unwrap();
            }
        }
    }
    let report = segment
        .payload_index
        .borrow()
        .audit_filter_indexes(&full_policy(), Some(1), &stopped())
        .unwrap();
    let audit = report.fields.get(&field).unwrap();
    assert!(
        audit
            .divergences
            .iter()
            .any(|d| d.kind == DivergenceKind::NullComplementMismatch && d.point_id == Some(0))
    );

    // Repair rebuilds the field including the null index.
    let plan = RepairPlan::from_report(&report);
    segment
        .payload_index
        .borrow_mut()
        .repair_filter_indexes(&plan, SegmentLifecycle::Unpublished, &stopped())
        .unwrap();
    let clean = segment
        .payload_index
        .borrow()
        .audit_filter_indexes(&full_policy(), Some(1), &stopped())
        .unwrap();
    assert!(clean.is_clean());
}

/// The repair must rewrite only the fields named in the plan: index files of other fields keep
/// their byte content, while the repaired field's files change and the re-audit is clean.
#[test]
fn repair_only_rewrites_planned_fields() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    let dir = Builder::new().prefix("audit_scope").tempdir().unwrap();
    let _scope = ambient::test_guard();
    let mut segment = build_simple_segment(dir.path(), 4, Distance::Dot).unwrap();
    let good_field: JsonPath = "good".parse().unwrap();
    let bad_field: JsonPath = "bad".parse().unwrap();

    for i in 0..4u64 {
        segment
            .upsert_point(i + 1, i.into(), only_default_vector(&[0.; 4]))
            .unwrap();
        let payload: Payload = serde_json::from_value(serde_json::json!({
            "good": i,
            "bad": i * 2,
        }))
        .unwrap();
        segment
            .set_full_payload(i + 10, i.into(), &payload)
            .unwrap();
    }
    segment
        .create_field_index(100, &good_field, Some(&PayloadSchemaType::Integer.into()))
        .unwrap();
    segment
        .create_field_index(101, &bad_field, Some(&PayloadSchemaType::Integer.into()))
        .unwrap();
    segment.flush(true).unwrap();

    // Hash every index file belonging to a field.
    let field_files = |field: &JsonPath| -> Vec<std::path::PathBuf> {
        segment.payload_index.borrow().field_indexes[field]
            .iter()
            .flat_map(|index| index.files())
            .collect()
    };
    let hash_files = |files: &[std::path::PathBuf]| -> Vec<u64> {
        files
            .iter()
            .map(|path| {
                let mut hasher = DefaultHasher::new();
                hasher.write(&fs_err::read(path).unwrap());
                hasher.finish()
            })
            .collect()
    };

    let good_files = field_files(&good_field);
    let good_before = hash_files(&good_files);

    // Corrupt only the bad field and persist the corruption, so the on-disk state the repair
    // starts from really differs from the freshly rebuilt state.
    {
        let mut index = segment.payload_index.borrow_mut();
        let indexes = index.field_indexes.get_mut(&bad_field).unwrap();
        for field_index in indexes.iter_mut() {
            if matches!(field_index, FieldIndex::NullIndex(_)) {
                continue;
            }
            field_index.remove_point(0).unwrap();
            field_index.flusher()().unwrap();
        }
    }
    let bad_corrupted = hash_files(&field_files(&bad_field));

    let report = segment
        .payload_index
        .borrow()
        .audit_filter_indexes(&full_policy(), Some(1), &stopped())
        .unwrap();
    assert_eq!(
        RepairPlan::from_report(&report).fields(),
        vec![bad_field.clone()],
        "only the corrupted field must enter the plan"
    );
    let plan = RepairPlan::from_report(&report);
    segment
        .payload_index
        .borrow_mut()
        .repair_filter_indexes(&plan, SegmentLifecycle::Unpublished, &stopped())
        .unwrap();
    segment.payload_index.borrow().flusher()().unwrap();

    // The good field's files are byte-identical; the bad field's differ from the corrupted state.
    assert_eq!(
        hash_files(&field_files(&good_field)),
        good_before,
        "fields outside the plan must not be rewritten"
    );
    assert_ne!(
        hash_files(&field_files(&bad_field)),
        bad_corrupted,
        "the planned field must be rewritten away from its corrupted state"
    );

    let clean = segment
        .payload_index
        .borrow()
        .audit_filter_indexes(&full_policy(), Some(1), &stopped())
        .unwrap();
    assert!(clean.is_clean());
}

/// A field the durable config lists but with no loaded index is a fatal MissingIndex.
#[test]
fn missing_loaded_index_is_fatal() {
    let dir = Builder::new().prefix("audit_missing").tempdir().unwrap();
    let field: JsonPath = "n".parse().unwrap();
    let segment = segment_with_payload(
        &dir,
        3,
        r#"{ "n": {id} }"#,
        &field,
        PayloadSchemaType::Integer.into(),
    );

    // Simulate a failed load: the config still lists the field, but no index is in memory.
    segment
        .payload_index
        .borrow_mut()
        .field_indexes
        .remove(&field);

    let report = segment
        .payload_index
        .borrow()
        .audit_filter_indexes(&full_policy(), Some(1), &stopped())
        .unwrap();
    let audit = report.fields.get(&field).unwrap();
    assert!(
        audit
            .divergences
            .iter()
            .any(|d| d.kind == DivergenceKind::MissingIndex)
    );
    assert_eq!(audit.highest_severity, Severity::Fatal);
}

/// Property-style check: with `k` injected divergences the report counts exactly `k` per corrupted
/// index, repair brings the count to zero, and a budget of one truncates the run.
#[test]
fn injected_divergence_count_and_budget_property() {
    let _scope = ambient::test_guard();
    let dir = Builder::new().prefix("audit_prop").tempdir().unwrap();
    let field: JsonPath = "color".parse().unwrap();
    let segment = segment_with_payload(
        &dir,
        10,
        r#"{ "color": "c-{id}" }"#,
        &field,
        PayloadSchemaType::Keyword.into(),
    );

    // Inject exactly three lost postings.
    let corrupted = [1u32, 4, 7];
    {
        let mut index = segment.payload_index.borrow_mut();
        let indexes = index.field_indexes.get_mut(&field).unwrap();
        for field_index in indexes.iter_mut() {
            if matches!(field_index, FieldIndex::NullIndex(_)) {
                continue;
            }
            for point_id in corrupted {
                field_index.remove_point(point_id).unwrap();
            }
        }
    }

    let report = segment
        .payload_index
        .borrow()
        .audit_filter_indexes(&full_policy(), Some(1), &stopped())
        .unwrap();
    let audit = report.fields.get(&field).unwrap();
    let missing: Vec<_> = audit
        .divergences
        .iter()
        .filter(|d| d.kind == DivergenceKind::MissingPosting)
        .map(|d| d.point_id)
        .collect();
    assert_eq!(missing, corrupted.into_iter().map(Some).collect::<Vec<_>>());

    // Repair and re-audit: zero divergences.
    let plan = RepairPlan::from_report(&report);
    segment
        .payload_index
        .borrow_mut()
        .repair_filter_indexes(&plan, SegmentLifecycle::Unpublished, &stopped())
        .unwrap();
    let clean = segment
        .payload_index
        .borrow()
        .audit_filter_indexes(&full_policy(), Some(1), &stopped())
        .unwrap();
    assert!(clean.is_clean());

    // A point budget of one truncates even Full mode.
    let tight = AuditPolicy {
        mode: AuditMode::Full,
        sample_size: 0,
        point_budget: 1,
        max_divergences: 8,
    };
    let truncated = segment
        .payload_index
        .borrow()
        .audit_filter_indexes(&tight, Some(1), &stopped())
        .unwrap();
    assert!(truncated.truncated);
    assert_eq!(truncated.checked_points, 1);
}

/// A mismatch between the config's recorded index types and the loaded indexes is a fatal
/// IndexTypeMismatch, reported separately from a missing index.
#[test]
fn index_type_mismatch_is_fatal() {
    let _scope = ambient::test_guard();
    let dir = Builder::new()
        .prefix("audit_type_mismatch")
        .tempdir()
        .unwrap();
    let field: JsonPath = "n".parse().unwrap();
    let segment = segment_with_payload(
        &dir,
        3,
        r#"{ "n": {id} }"#,
        &field,
        PayloadSchemaType::Integer.into(),
    );

    // Drop one loaded index while the durable config still records it: loaded types no longer
    // correspond one-to-one to the config's list.
    {
        let mut index = segment.payload_index.borrow_mut();
        let indexes = index.field_indexes.get_mut(&field).unwrap();
        let position = indexes
            .iter()
            .position(|candidate| !matches!(candidate, FieldIndex::NullIndex(_)))
            .unwrap();
        indexes.remove(position);
    }

    let report = segment
        .payload_index
        .borrow()
        .audit_filter_indexes(&full_policy(), Some(1), &stopped())
        .unwrap();
    let audit = report.fields.get(&field).unwrap();
    assert!(
        audit
            .divergences
            .iter()
            .any(|d| d.kind == DivergenceKind::IndexTypeMismatch)
    );
    assert_eq!(audit.highest_severity, Severity::Fatal);
}
