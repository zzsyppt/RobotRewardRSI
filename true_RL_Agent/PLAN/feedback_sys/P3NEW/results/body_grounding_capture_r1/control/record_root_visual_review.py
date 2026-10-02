"""Record root's completed original-pixel development review; never score an observer."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[1]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

verification = json.loads((root / "receipt/verification.json").read_bytes())
notes = {
    1: "Full body is in frame in both views. The side view already shows angled leg segments; front-view depth is foreshortened. Palm-origin projections locate wrist-side attachments, not palm centres.",
    44: "Left palm origin matches the hand brought across the lower-front body in v0 and the forward wrist in v1. The other hand in v1 is right. The leg configuration has changed locally, without evidence of a new whole-body squat category.",
    71: "Left wrist is forward and higher in the side image than at step 44; its projection follows that location. Near/far leg names remain consistent between views. No continuous motion is inferred between samples.",
    109: "Projected left wrist returns to a lower image position than at 71, while the right hand is at the opposite side. This is a sampled-state comparison, not proof of a repeated action cycle.",
    206: "Left wrist projection is on the forward arm in v1. Right wrist is close to the trunk projection and partly occluded in that view; v0 separates it. Neither image proves surface contact or support force.",
    412: "Both full-body views remain usable. Left wrist is lower in the side image than at 206; right wrist overlaps the trunk region in v1 and is better disambiguated in v0. Knee/ankle projections follow the respective leg regions.",
}
images = []
for row in verification["runs"][0]["images"]:
    image = root / "materials/push_run01/video" / Path(row["file"]).name
    assert sha(image) == row["sha256"]
    images.append({"file": image.relative_to(root).as_posix(), "sha256": row["sha256"],
        "step": row["control_step"], "view_id": row["view_id"], "root_viewed_original_pixels": True,
        "part_localization": "coarsely_consistent_with_named_link_origins",
        "note": notes[row["control_step"]]})
assert len(images) == 12
result = {"schema_version": "p3new.body_grounding_pixel_review.v1",
    "reviewer": {"id": "codex-root-body-grounding-r1", "kind": "development_model_review"},
    "status": "PASS_FOR_OLD_DEVELOPMENT_BODY_LOCALIZATION",
    "source_receipts": {str(path.relative_to(root)): sha(path) for path in (
        root / "control/capture_card.json", root / "receipt/verification.json",
        root / "materials/run01_body_geometry.json", root / "materials/run01_camera_origin_check.json")},
    "images": images,
    "scope": "12 original images, run1/env19, six fixed actual steps, two views; no new observer output",
    "findings": ["Full body and scene reference are present; no obvious camera/view swap or mirror error observed.",
        "Calibrated link points help identify left/right parts without treating projection as visibility.",
        "Step1 to44 H-K-A angle decreases on the left and increases on the right; sides must remain independent.",
        "The deterministic geometry is not an anatomical knee angle, whole-body crouch score, or causal diagnosis."],
    "limits": ["Interior link origins can lie behind meshes; pixels do not independently expose every exact joint centre.",
        "No independently annotated pixel coordinates or subpixel error bound were obtained.",
        "Only 12 of the 32 new PNGs were manually viewed; all32 were mechanically checked.",
        "This is nonblind old-development engineering review, not N5 or raw-observer accuracy acceptance.",
        "No API call tested whether the observer uses the new evidence correctly; historical r10 errors remain unchanged."],
    "provider_requests": 0}
out = root / "receipt/root_visual_review.json"
with out.open("x", encoding="utf-8") as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps({"status": result["status"], "reviewed_images": len(images), "sha256": sha(out)}))
