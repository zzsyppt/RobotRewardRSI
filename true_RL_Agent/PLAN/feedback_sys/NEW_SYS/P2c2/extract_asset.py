"""CPU Sdf only: read two SHA-locked layers, print authored facts; no writes."""
import hashlib
import json
from pathlib import Path
import sys
from pxr import Sdf, Usd

EXPECTED = ["ba171c972b987d8c8fb7157ccad2ba9c0c1fed105755d2d8af46bef96cc11c6d",
            "2902646d0f4c13c9ecae3ac9046e32c7d18902eef9679de9829f642a15c93fb5"]


def value(v):
    if hasattr(v, "GetReal"):
        return [v.GetReal(), *v.GetImaginary()]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return [str(x) if isinstance(x, Sdf.Path) else x for x in v]


layers, joints, references = [], [], []
for index, filename in enumerate(sys.argv[1:]):
    path = Path(filename)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == EXPECTED[index]
    layer = Sdf.Layer.FindOrOpen(str(path))
    assert not layer.subLayerPaths
    layers.append({"sha256": digest, "bytes": path.stat().st_size,
                   "external_references": sorted(layer.GetExternalReferences())})
    def visit(p):
        prim = layer.GetPrimAtPath(p)
        if not prim:
            return
        refs = prim.referenceList.GetAppliedItems()
        for ref in refs:
            references.append({"layer": index, "prim": str(p), "asset_path": ref.assetPath,
                               "target_prim": str(ref.primPath)})
        if prim.typeName == "PhysicsRevoluteJoint":
            assert index == 0
            fields = {name: value(prim.attributes[name].default) for name in
                      ("physics:axis", "physics:localPos0", "physics:localPos1", "physics:localRot0",
                       "physics:localRot1", "physics:lowerLimit", "physics:upperLimit")}
            bodies = {name: str(prim.relationships[name].targetPathList.GetAppliedItems()[0])
                      for name in ("physics:body0", "physics:body1")}
            fields.update(bodies)
            fields["body_authored_transforms"] = {
                name: {key: value(layer.GetPrimAtPath(body).attributes[key].default)
                       for key in ("xformOp:orient", "xformOp:translate", "xformOp:scale")}
                for name, body in bodies.items()}
            joints.append({"name": prim.name, "prim": str(p), **fields})
    layer.Traverse("/", visit)
    if index == 0:
        root_usda = layer.ExportToString()
assert len(layers) == 2 and layers[1]["external_references"] == []
assert layers[0]["external_references"] == ["./Props/instanceable_meshes.usd"]
mesh = Sdf.Layer.FindOrOpen(sys.argv[2])
assert all(r["layer"] == 0 and r["asset_path"] == "./Props/instanceable_meshes.usd"
           and mesh.GetPrimAtPath(r["target_prim"]) for r in references)
print(json.dumps({"schema_version": "go2.nominal_usd_extraction.v1", "usd_version": list(Usd.GetVersion()),
    "layers": layers, "references": references, "joints": joints, "root_usda": root_usda}, indent=2))
