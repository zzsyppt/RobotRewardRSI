"""Independent composed USD audit on CPU; anonymous remap only, no file writes."""
import hashlib
import json
from pathlib import Path
import sys
import resource
import time
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, Gf

started, cpu = time.perf_counter(), time.process_time()

root_path, mesh_path = map(Path, sys.argv[1:])
hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (root_path, mesh_path)]
assert hashes == ["ba171c972b987d8c8fb7157ccad2ba9c0c1fed105755d2d8af46bef96cc11c6d",
                  "2902646d0f4c13c9ecae3ac9046e32c7d18902eef9679de9829f642a15c93fb5"]
root = Sdf.Layer.OpenAsAnonymous(str(root_path))
refs = []
def remap(path):
    prim = root.GetPrimAtPath(path)
    if not prim: return
    original = prim.referenceList.GetAppliedItems()
    if original:
        assert all(r.assetPath == "./Props/instanceable_meshes.usd" for r in original)
        refs.extend(str(r.primPath) for r in original)
        prim.referenceList.ClearEdits()
        prim.referenceList.prependedItems = [Sdf.Reference(str(mesh_path), r.primPath) for r in original]
root.Traverse("/", remap)
stage = Usd.Stage.Open(root)
cache = UsdGeom.XformCache()
rows = []
for prim in stage.Traverse():
    if not prim.IsA(UsdPhysics.RevoluteJoint): continue
    joint = UsdPhysics.RevoluteJoint(prim)
    bodies = [joint.GetBody0Rel().GetTargets()[0], joint.GetBody1Rel().GetTargets()[0]]
    rotations = [joint.GetLocalRot0Attr().Get(), joint.GetLocalRot1Attr().Get()]
    positions = [joint.GetLocalPos0Attr().Get(), joint.GetLocalPos1Attr().Get()]
    transforms = [cache.GetLocalToWorldTransform(stage.GetPrimAtPath(b)) for b in bodies]
    global_rotations = [t.ExtractRotationQuat()*Gf.Quatd(r) for t, r in zip(transforms, rotations)]
    vectors = [[q.GetReal(), *q.GetImaginary()] for q in global_rotations]
    assert max(abs(a-b) for a, b in zip(*vectors)) < 1e-7
    anchors = [t.Transform(Gf.Vec3d(p)) for t, p in zip(transforms, positions)]
    residual = (anchors[0]-anchors[1]).GetLength()
    assert residual < 1e-7
    axis = joint.GetAxisAttr().Get()
    assert axis == "X"
    rotated = rotations[0].GetNormalized().Transform(Gf.Vec3f(1, 0, 0))
    rows.append({"name": prim.GetName(), "prim": str(prim.GetPath()), "body0": str(bodies[0]), "body1": str(bodies[1]),
                 "axis": axis, "axis_body0": list(rotated), "lower_deg": joint.GetLowerLimitAttr().Get(),
                 "upper_deg": joint.GetUpperLimitAttr().Get(), "zero_rotation_equal": True,
                 "zero_anchor_residual_m": residual})
assert len(rows) == 12 and len(refs) == 36
assert all(stage.GetPrimAtPath(p).IsValid() for p in refs)
print(json.dumps({"status": "PASS", "scope": "composed_original_SHA_locked_layers_anonymous_reference_remap_CPU",
    "original_asset_sha256": hashes, "resolved_reference_count": len(refs), "joints": rows,
    "used_layers": sorted(l.identifier for l in stage.GetUsedLayers()), "server_writes": 0,
    "wall_seconds": time.perf_counter()-started, "cpu_seconds": time.process_time()-cpu,
    "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}, indent=2))
