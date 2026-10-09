"""Deterministic references and report text coverage; never semantic scores."""
from __future__ import annotations

import hashlib
import json
import re

REPORT_TEXT_ROLES = {"automatic_report", "full_derivation", "bounded_request", "observer_output"}
SHARED_BINDING_FIELDS = {"domain", "binding", "comparison_domain", "source_bindings"}
ANCHOR_CONTRACT = "p3new.reference_anchors.v1"
BLOCK_CONTRACT = "p3new.reference_blocks.v2"
_UNSPLIT_ROLES = {"raw_image", "review_protocol", "focus_questions", "own_sealed_facts"}
BLOCK_INSTRUCTION = (
    "Each reference_material_block places its anchor_id and original locator beside its complete value. "
    "Copy the supplied anchor_id as a JSON string only; do not construct wrappers or block-local JSON pointers. "
    "Read the complete supplied value. Material headers retain original "
    "envelope metadata; data_canonical_sha256 verifies reconstructed canonical JSON data, not original file bytes. "
    "The reference_catalog retains additional exact report text anchors, including registered nested JSON strings."
)
REFERENCE_INSTRUCTION = (
    "Every evidence_refs entry identifies exactly one supplied material as {material_id,sha256,json_pointer}. "
    "The JSON pointer starts at its supplied envelope (therefore material data begins at /data); "
    "image metadata uses its envelope directly. Never use brace unions, wildcards, or external archive paths. "
)
INSTRUCTION = (
    "The report-only review_text_inventory lists original explanatory text, not truth labels or atomic claims. "
    "Return findings.inventory_review with exactly one entry per inventory_id: status reviewed or unreviewed, "
    "claim_ids, and a nonempty reason. Reviewed entries must reference claims whose report_locator is the exact "
    "inventory locator; split composite text into all independently falsifiable propositions yourself. "
    "Unreviewed entries keep complete=false. Each claim needs a verdict, reason, report_locator (or report_locators), and evidence_refs. "
    "Each fact_matches row must use the output key sealed_fact_id, copied exactly from the corresponding "
    "own sealed necessary_facts.fact_id; include every ID exactly once. Its status is matched, missing or disputed. "
    "A merely partial match is missing with the covered and missing parts explained in reason; never shrink the denominator. "
    "A locator is {material_id,sha256,json_pointer}, relative to the supplied material envelope; "
    "for a nested JSON string, first use decode_json_string_at and then json_pointer inside that decoded value. "
    "Never use brace unions, wildcard paths, or undocumented original-archive paths as supplied-material pointers. "
    "Formal numeric instances remain the separate arithmetic responsibility; inspect their meanings and relations. "
    "ABI, input availability and permission clauses require independent supplied task_definition support; "
    "the report, its inventory, or arithmetic agreement alone cannot establish those claims. If not independently "
    "established, record unsupported or unanswerable and the exact source gap, not supported. "
    "Contradicted, unsupported and unanswerable are completed judgments; unreviewed means genuinely not assessed. "
    "Mechanical coverage and resolvable references do not prove semantic support or exhaustive proposition splitting."
)
ANCHOR_INSTRUCTION = (
    "Select evidence references only from the supplied reference_catalog. evidence_refs is a JSON array of "
    "complete anchor_id strings copied exactly from that catalog, for example [\"a-...\",\"a-...\"]. "
    "Do not wrap an ID in an object. Never output within_anchor_pointer, material_id, sha256, json_pointer, "
    "external paths, wildcards or brace unions. The anchor cites its whole registered block. "
    "Keep the relevant identity, body, steps or intervals and scope explicit in the original fact/claim fields; "
    "a block reference does not broaden the stated scope. Retain the independent arithmetic responsibility. "
    "Return the supplied top-level reference_catalog_sha256. A resolvable anchor establishes only location, "
    "not semantic support; keep missing evidence and uncertainty explicit."
)
ANCHOR_REPORT_INSTRUCTION = INSTRUCTION.replace(
    "A locator is {material_id,sha256,json_pointer}, relative to the supplied material envelope; "
    "for a nested JSON string, first use decode_json_string_at and then json_pointer inside that decoded value. ",
    "report_locator is the exact supplied text-* inventory_id JSON string; report_locators, when needed, "
    "is an array of those exact strings. Do not construct objects or paths. "
    "their normalized locators must exactly equal the original "
    "inventory locators, including any registered nested-JSON decoding. Do not reconstruct decoding paths. "
)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def pointer(value, path):
    """RFC 6901 object/array lookup; no wildcard, union, negative index or coercion."""
    if not isinstance(path, str) or (path and not path.startswith("/")):
        raise ValueError("JSON pointer must be empty or begin with /")
    for part in path.split("/")[1:]:
        if re.search(r"~(?![01])", part):
            raise ValueError("invalid JSON pointer escape")
        key = part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", key):
                raise ValueError("invalid array index")
            value = value[int(key)]
        elif isinstance(value, dict):
            value = value[key]
        else:
            raise ValueError("pointer traverses scalar")
    return value


def _structural_blocks(value, locator, depth=0):
    """The catalog and presentation share one two-level, dict-only frontier."""
    if isinstance(value, dict) and value and depth < 2:
        for key, child in value.items():
            escaped = key.replace("~", "~0").replace("/", "~1")
            yield from _structural_blocks(child,
                {**locator, "json_pointer": locator["json_pointer"] + "/" + escaped}, depth + 1)
    else:
        yield locator, value


def _catalog_anchors(catalog):
    if not isinstance(catalog, dict) or catalog.get("schema_version") != ANCHOR_CONTRACT or not isinstance(catalog.get("items"), list):
        raise ValueError("invalid reference catalog contract")
    by_locator, identities = {}, set()
    for item in catalog["items"]:
        if not isinstance(item, dict) or set(item) != {"anchor_id", "locator"}:
            raise ValueError("invalid reference catalog entry")
        identity, locator = item["anchor_id"], item["locator"]
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError("duplicate or invalid reference anchor identity")
        if not isinstance(locator, dict) or not {"material_id", "sha256", "json_pointer"} <= set(locator) <= {"material_id", "sha256", "json_pointer", "decode_json_string_at"}:
            raise ValueError("invalid reference anchor locator")
        address = canonical(locator)
        if address in by_locator:
            raise ValueError("duplicate reference anchor locator")
        identities.add(identity)
        by_locator[address] = identity
    return by_locator


def render_material_blocks(material_envelope, catalog):
    """Present unchanged JSON values beside their existing, exact catalog anchors."""
    def text_block(value):
        # Preserve insertion order for presentation; canonical hashing is separate.
        return {"type": "text", "text": json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)}

    if material_envelope.get("role") in _UNSPLIT_ROLES:
        # Match the legacy runner's text serialization for untouched messages,
        # especially the raw-image label immediately preceding its PNG block.
        return [{"type": "text", "text": json.dumps(material_envelope, ensure_ascii=False, allow_nan=False)}]
    anchors = _catalog_anchors(catalog)
    data = material_envelope["data"]
    base = {"material_id": material_envelope["material_id"], "sha256": material_envelope["sha256"], "json_pointer": "/data"}
    leaves = list(_structural_blocks(data, base))
    blocks = [text_block({"schema_version": BLOCK_CONTRACT, "role": "reference_material_header",
        "envelope": {key: value for key, value in material_envelope.items() if key != "data"},
        "data_canonical_sha256": hashlib.sha256(canonical(data)).hexdigest(), "block_count": len(leaves)})]
    for locator, value in leaves:
        if canonical(locator) not in anchors:
            raise ValueError("material block missing from reference catalog")
        blocks.append(text_block({"schema_version": BLOCK_CONTRACT, "role": "reference_material_block",
            "anchor_id": anchors[canonical(locator)], "locator": locator, "value": value}))
    return blocks


def _restore_material(header, blocks):
    """Rebuild only the registered structural partition; never infer missing paths."""
    if set(header) != {"schema_version", "role", "envelope", "data_canonical_sha256", "block_count"} or header.get("schema_version") != BLOCK_CONTRACT:
        raise ValueError("invalid reference material header")
    envelope = header["envelope"]
    if not isinstance(envelope, dict) or not {"material_id", "role", "sha256"} <= set(envelope) or "data" in envelope or envelope.get("role") in _UNSPLIT_ROLES:
        raise ValueError("invalid split material envelope")
    if type(header["block_count"]) is not int or header["block_count"] < 1 or len(blocks) != header["block_count"]:
        raise ValueError("missing or duplicate material blocks")
    data, paths, identities, locators = {}, [], set(), []
    for block in blocks:
        if set(block) != {"schema_version", "role", "anchor_id", "locator", "value"} or block.get("schema_version") != BLOCK_CONTRACT:
            raise ValueError("invalid reference material block")
        identity, locator = block["anchor_id"], block["locator"]
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError("duplicate or invalid material block anchor")
        identities.add(identity)
        if not isinstance(locator, dict) or set(locator) != {"material_id", "sha256", "json_pointer"} or locator["material_id"] != envelope["material_id"] or locator["sha256"] != envelope["sha256"]:
            raise ValueError("material block locator differs from envelope")
        path = locator["json_pointer"]
        if not isinstance(path, str) or not (path == "/data" or path.startswith("/data/")) or re.search(r"~(?![01])", path):
            raise ValueError("invalid structural material block pointer")
        parts = tuple(part.replace("~1", "/").replace("~0", "~") for part in path.split("/")[2:])
        if len(parts) > 2:
            raise ValueError("material block is outside structural frontier")
        if any(parts[:len(old)] == old or old[:len(parts)] == parts for old in paths):
            raise ValueError("overlapping material blocks")
        paths.append(parts)
        locators.append(locator)
        if not parts:
            data = block["value"]
        else:
            parent = data
            for key in parts[:-1]:
                parent = parent.setdefault(key, {})
            parent[parts[-1]] = block["value"]
    base = {"material_id": envelope["material_id"], "sha256": envelope["sha256"], "json_pointer": "/data"}
    if locators != [locator for locator, _ in _structural_blocks(data, base)]:
        raise ValueError("material blocks differ from ordered structural frontier")
    if hashlib.sha256(canonical(data)).hexdigest() != header["data_canonical_sha256"]:
        raise ValueError("reconstructed material canonical data hash differs")
    return {**envelope, "data": data}


def _unique_json_object(pairs):
    value = {}
    for key, child in pairs:
        if key in value:
            raise ValueError("duplicate JSON object key in material blocks")
        value[key] = child
    return value


def _invalid_json_constant(value):
    raise ValueError("non-finite JSON number in material blocks: " + value)


def request_materials(request):
    materials = {}
    inventory = None
    split = {}
    for message in request["messages"]:
        if not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if block.get("type") != "text":
                continue
            try:
                item = json.loads(block["text"])
            except (ValueError, TypeError):
                continue
            if not isinstance(item, dict):
                continue
            if item.get("role") in {"reference_material_header", "reference_material_block"}:
                item = json.loads(block["text"], object_pairs_hook=_unique_json_object,
                                  parse_constant=_invalid_json_constant)
            if item.get("role") == "reference_material_header":
                envelope = item.get("envelope")
                if not isinstance(envelope, dict) or not isinstance(envelope.get("material_id"), str):
                    raise ValueError("invalid split material envelope")
                identity = envelope["material_id"]
                if identity in materials:
                    raise ValueError("duplicate material identity")
                materials[identity] = envelope
                split[identity] = (item, [])
            elif item.get("role") == "reference_material_block":
                locator = item.get("locator")
                if not isinstance(locator, dict) or not isinstance(locator.get("material_id"), str) or locator["material_id"] not in split:
                    raise ValueError("material block has no preceding header")
                split[locator["material_id"]][1].append(item)
            elif item.get("role") == "review_text_inventory":
                if inventory is not None:
                    raise ValueError("duplicate inventory")
                inventory = item["data"]
            elif "material_id" in item:
                if item["material_id"] in materials:
                    raise ValueError("duplicate material identity")
                materials[item["material_id"]] = item
            elif item.get("role") == "own_sealed_facts":
                materials["own_sealed_facts"] = item
    if split:
        anchors = _catalog_anchors(request_reference_catalog(request))
        for identity, (header, blocks) in split.items():
            materials[identity] = _restore_material(header, blocks)
            for block in blocks:
                if anchors.get(canonical(block["locator"])) != block["anchor_id"]:
                    raise ValueError("material block anchor differs from reference catalog")
    return materials, inventory


def resolve_reference(reference, materials):
    if not isinstance(reference, dict):
        raise ValueError("reference must identify one material and pointer")
    material = materials[reference["material_id"]]
    if reference["sha256"] != material["sha256"]:
        raise ValueError("reference material hash differs")
    value = material
    if "decode_json_string_at" in reference:
        value = json.loads(pointer(value, reference["decode_json_string_at"]))
    return pointer(value, reference["json_pointer"])


def build_reference_catalog(materials, inventory=None):
    """Bounded structural addresses only; never select or label scientific facts."""
    items, identities = [], set()
    inventory_items = [] if inventory is None else inventory["items"]
    inventory_locators = {canonical(item["locator"]) for item in inventory_items}

    def add(locator, anchor_id=None):
        resolve_reference(locator, materials)
        if anchor_id is None and canonical(locator) in inventory_locators:
            return  # The existing text inventory owns this exact address.
        identity = anchor_id if anchor_id is not None else "a-" + hashlib.sha256(canonical(locator)).hexdigest()[:16]
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError("duplicate or invalid reference anchor identity")
        identities.add(identity)
        items.append({"anchor_id": identity, "locator": dict(locator)})

    excluded = {"review_protocol", "focus_questions", "own_sealed_facts"}
    for material_id in sorted(materials):
        material = materials[material_id]
        if material.get("role") in excluded:
            continue
        base = {"material_id": material_id, "sha256": material["sha256"], "json_pointer": ""}
        if material.get("role") == "raw_image":
            add(base)
        else:
            for locator, _ in _structural_blocks(material["data"], {**base, "json_pointer": "/data"}):
                add(locator)
    for item in inventory_items:
        if materials[item["locator"]["material_id"]].get("role") not in REPORT_TEXT_ROLES:
            raise ValueError("inventory anchor points outside report materials")
        add(item["locator"], item["inventory_id"])
    return {"schema_version": ANCHOR_CONTRACT, "items": sorted(items, key=lambda item: item["anchor_id"])}


def request_reference_catalog(request):
    """Read exactly one optional catalog envelope without treating it as evidence."""
    catalog = None
    found = False
    for message in request["messages"]:
        if not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if block.get("type") != "text":
                continue
            try:
                item = json.loads(block["text"])
            except (ValueError, TypeError):
                continue
            if not isinstance(item, dict) or item.get("role") != "reference_catalog":
                continue
            if found:
                raise ValueError("duplicate reference catalog")
            found = True
            catalog = item.get("data")
            if not isinstance(catalog, dict):
                raise ValueError("reference catalog must be an object")
    return catalog


def normalize_reference(reference, materials, catalog):
    """Resolve an exact ID string or historical object/pointer; never repair."""
    if isinstance(reference, str):
        reference = {"anchor_id": reference}
    if not isinstance(reference, dict) or not {"anchor_id"} <= set(reference) <= {"anchor_id", "within_anchor_pointer"}:
        raise ValueError("anchor reference permits only anchor_id and within_anchor_pointer")
    if not isinstance(reference["anchor_id"], str) or not reference["anchor_id"]:
        raise ValueError("anchor_id must be a nonempty string")
    if not isinstance(catalog, dict) or catalog.get("schema_version") != ANCHOR_CONTRACT or not isinstance(catalog.get("items"), list):
        raise ValueError("invalid reference catalog contract")
    by_id = {}
    for item in catalog["items"]:
        if not isinstance(item, dict) or set(item) != {"anchor_id", "locator"}:
            raise ValueError("invalid reference catalog entry")
        identity = item["anchor_id"]
        if not isinstance(identity, str) or not identity or identity in by_id:
            raise ValueError("duplicate or invalid reference anchor identity")
        by_id[identity] = item["locator"]
    if reference["anchor_id"] not in by_id:
        raise ValueError("unknown reference anchor")
    locator = by_id[reference["anchor_id"]]
    if not isinstance(locator, dict) or not {"material_id", "sha256", "json_pointer"} <= set(locator) <= {"material_id", "sha256", "json_pointer", "decode_json_string_at"}:
        raise ValueError("invalid reference anchor locator")
    within = reference.get("within_anchor_pointer", "")
    # Resolve within the registered subtree first; CSV strings remain scalars.
    pointer(resolve_reference(locator, materials), within)
    normalized = {**locator, "json_pointer": locator["json_pointer"] + within}
    resolve_reference(normalized, materials)
    return normalized


def text_roots(materials):
    """Explicit current report schemas; original materials remain completely supplied."""
    roots = []
    linked_requests = set()
    for material_id, item in materials.items():
        if item.get("role") != "automatic_report" or "request_material_id" not in item.get("data", {}):
            continue
        locator = item["data"]
        target = materials[locator["request_material_id"]]
        if target["role"] != "bounded_request" or target["sha256"] != locator["request_sha256"]:
            raise ValueError("bounded report locator request differs")
        roots.append({"material_id": locator["request_material_id"], "sha256": target["sha256"],
            "decode_json_string_at": "/data" + locator["decode_json_string_at"],
            "json_pointer": locator["then_json_pointer"] + "/report"})
        linked_requests.add(locator["request_material_id"])
    for material_id, item in materials.items():
        role, data = item.get("role"), item.get("data")
        if role not in REPORT_TEXT_ROLES:
            continue
        base = {"material_id": material_id, "sha256": item["sha256"]}
        explicit = item.get("review_text_roots")
        if explicit is not None:
            roots.extend({**base, **root} for root in explicit)
        elif role == "full_derivation":
            if not isinstance(data, dict) or "shared" not in data:
                raise ValueError("full derivation requires explicit explanatory text roots")
            roots.extend({**base, "json_pointer": "/data/shared/" + field.replace("~", "~0").replace("/", "~1")}
                         for field in data["shared"] if field not in SHARED_BINDING_FIELDS)
        elif role == "automatic_report":
            if "request_material_id" not in data:
                if "report" not in data:
                    raise ValueError("automatic report requires report tree or explicit text roots")
                roots.append({**base, "json_pointer": "/data/report"})
        elif role == "bounded_request" and material_id not in linked_requests:
            raise ValueError("bounded request requires exact report locator or explicit text roots")
        elif role == "observer_output":
            raise ValueError("observer output requires explicit original observation text roots")
    return roots


def build_inventory(materials):
    roots, items = text_roots(materials), []
    def walk(value, reference):
        if isinstance(value, dict):
            for key, child in value.items():
                # Sentence identifiers are locators, not additional semantic claims.
                if key in {"id", "schema_version", "version", "lexicon_version", "sampling_seed", "hash_seed"}:
                    continue
                walk(child, {**reference, "json_pointer": reference["json_pointer"] + "/" + key.replace("~", "~0").replace("/", "~1")})
        elif isinstance(value, list) and not (value and all(isinstance(v, str) for v in value)):
            for i, child in enumerate(value):
                walk(child, {**reference, "json_pointer": reference["json_pointer"] + "/" + str(i)})
        elif isinstance(value, str) or (isinstance(value, list) and value):
            identity = hashlib.sha256(canonical({"locator": reference, "original": value})).hexdigest()[:24]
            items.append({"inventory_id": "text-" + identity, "locator": reference, "original": value})
    for root in roots:
        walk(resolve_reference(root, materials), root)
    if len({item["inventory_id"] for item in items}) != len(items):
        raise ValueError("overlapping explanatory text roots")
    return {"schema_version": "p3new.report_text_inventory.v1", "items": items, "text_roots": roots,
        "scope": "Original explanatory text/term lists in declared shared definitions and actual report trees; lists preserve all original members. Composite claims still require reviewer splitting.",
        "separate_responsibilities": ["Complete episode formal fields/events and numeric leaves: independent arithmetic coverage, not one model claim per scalar",
            "Identity/hash fields: separate source binding, not automatically verified by this inventory",
            "Actual meanings, scope, relations, ABI/authority and necessary process coverage: model review; source availability alone is not support"],
        "semantic_support_proven": False}


def audit_answer(answer, request, *, stage):
    materials, supplied = request_materials(request)
    errors, references = [], []
    resolved_references = []
    catalog, anchor_mode, catalog_valid, expected_inventory = None, False, False, None
    try:
        catalog = request_reference_catalog(request)
        anchor_mode = catalog is not None
        if anchor_mode:
            expected_inventory = build_inventory(materials) if stage == "report" else None
            expected_catalog = build_reference_catalog(materials, expected_inventory)
            if canonical(catalog) != canonical(expected_catalog):
                errors.append("reference_catalog_does_not_match_actual_materials")
            else:
                catalog_valid = True
            if not isinstance(answer, dict) or answer.get("reference_catalog_sha256") != hashlib.sha256(canonical(catalog)).hexdigest():
                errors.append("answer_reference_catalog_sha256_missing_or_differs")
    except (ValueError, KeyError, IndexError, TypeError) as error:
        anchor_mode = True
        errors.append({"issue": "invalid_reference_catalog", "detail": str(error)})
    findings = answer.get("findings", {}) if isinstance(answer, dict) else {}
    if not isinstance(findings, dict):
        findings = {}
        errors.append("findings_not_object")
    def check_ref(ref, path, report=False):
        try:
            if anchor_mode:
                if not catalog_valid:
                    raise ValueError("reference catalog was not verified")
                ref = normalize_reference(ref, materials, catalog)
            elif isinstance(ref, str) and report:
                # Legacy single-material pointers are accepted only if unambiguous.
                possible = []
                for mid, item in materials.items():
                    if item.get("role") in REPORT_TEXT_ROLES:
                        candidate = {"material_id": mid, "sha256": item["sha256"], "json_pointer": ref}
                        try: resolve_reference(candidate, materials)
                        except (ValueError, KeyError, IndexError, TypeError): continue
                        possible.append(candidate)
                if len(possible) != 1: raise ValueError("legacy report pointer is absent or ambiguous")
                ref = possible[0]
            resolve_reference(ref, materials)
            if report and materials[ref["material_id"]].get("role") not in REPORT_TEXT_ROLES:
                raise ValueError("report locator points outside report materials")
            references.append({"answer_locator": path, "status": "resolves"})
            if anchor_mode:
                resolved_references.append({"answer_locator": path, "locator": ref})
            return ref
        except (ValueError, KeyError, IndexError, TypeError) as error:
            errors.append({"answer_locator": path, "issue": "unresolvable_reference", "detail": str(error)})
            return None
    def evidence_walk(value, path="/findings"):
        if isinstance(value, dict):
            for key, child in value.items():
                p = path + "/" + key
                if key == "evidence_refs":
                    if not isinstance(child, list): errors.append({"answer_locator": p, "issue": "expected_reference_list"})
                    else:
                        for i, ref in enumerate(child): check_ref(ref, p + "/" + str(i))
                elif key in {"report_locator", "report_locators"} and child is not None:
                    if not path.startswith("/findings/claims/"):
                        for i, ref in enumerate(child if isinstance(child, list) else [child]):
                            suffix = "/" + str(i) if isinstance(child, list) or not anchor_mode else ""
                            check_ref(ref, p + suffix, True)
                else: evidence_walk(child, p)
        elif isinstance(value, list):
            for i, child in enumerate(value): evidence_walk(child, path + "/" + str(i))
    evidence_walk(findings)
    if stage == "facts":
        if supplied is not None or any(m.get("role") in REPORT_TEXT_ROLES | {"arithmetic_audit"} for m in materials.values()):
            errors.append("facts_received_report_or_inventory")
        return {"stage": stage, "mechanical_valid": not errors, "errors": errors,
                "declared_complete": answer.get("complete"), "review_complete": not errors and answer.get("complete") is True,
                "references_checked": len(references), "semantic_support_proven": False,
                **({"resolved_references": resolved_references} if anchor_mode else {})}
    expected = expected_inventory if expected_inventory is not None else build_inventory(materials)
    if supplied != expected:
        errors.append("inventory_not_supplied_or_does_not_match_actual_materials")
    claims, claim_refs = {}, {}
    claim_rows = findings.get("claims", [])
    if not isinstance(claim_rows, list):
        errors.append("claims_not_list")
        claim_rows = []
    for i, claim in enumerate(claim_rows):
        if not isinstance(claim, dict) or not isinstance(claim.get("claim_id"), str) or claim["claim_id"] in claims:
            errors.append({"claim_index": i, "issue": "invalid_or_duplicate_claim_id"}); continue
        cid = claim["claim_id"]
        claims[cid] = claim
        locator_key = "report_locators" if "report_locators" in claim else "report_locator"
        locator = claim.get(locator_key)
        refs = locator if isinstance(locator, list) else [locator]
        claim_refs[cid] = [check_ref(ref,
            f"/findings/claims/{i}/{locator_key}" + (f"/{j}" if isinstance(locator, list) else "")
            if anchor_mode else f"/findings/claims/{i}/report_locator", True) for j, ref in enumerate(refs)]
        if not claim.get("verdict") or not claim.get("reason") or not isinstance(claim.get("evidence_refs"), list):
            errors.append({"claim_id": cid, "issue": "claim_judgment_or_evidence_record_missing"})
    inventory_ids = {item["inventory_id"]: item for item in expected["items"]}
    seen, unreviewed = set(), []
    review_rows = findings.get("inventory_review", [])
    if not isinstance(review_rows, list):
        errors.append("inventory_review_not_list")
        review_rows = []
    for row in review_rows:
        if not isinstance(row, dict): errors.append("invalid_inventory_review_entry"); continue
        iid = row.get("inventory_id")
        if iid not in inventory_ids or iid in seen:
            errors.append({"inventory_id": iid, "issue": "unknown_or_duplicate_inventory_entry"}); continue
        seen.add(iid)
        if row.get("status") not in {"reviewed", "unreviewed"} or not str(row.get("reason") or "").strip():
            errors.append({"inventory_id": iid, "issue": "explicit_status_and_reason_required"})
        if row.get("status") == "unreviewed":
            unreviewed.append(iid)
        elif not isinstance(row.get("claim_ids"), list) or not row["claim_ids"]:
            errors.append({"inventory_id": iid, "issue": "reviewed_item_requires_claim_ids"})
        else:
            for cid in row["claim_ids"]:
                if cid not in claims or inventory_ids[iid]["locator"] not in claim_refs[cid]:
                    errors.append({"inventory_id": iid, "claim_id": cid, "issue": "claim_must_point_to_exact_inventory_text"})
    missing = sorted(set(inventory_ids)-seen)
    if missing: errors.append({"issue": "inventory_entries_missing", "inventory_ids": missing})
    other_unreviewed = findings.get("unreviewed_locators", [])
    if not isinstance(other_unreviewed, list):
        errors.append("unreviewed_locators_not_list")
        other_unreviewed = ["invalid"]
    for i, value in enumerate(other_unreviewed):
        check_ref(value.get("locator", value) if isinstance(value, dict) else value,
                  f"/findings/unreviewed_locators/{i}" + ("/locator" if anchor_mode and isinstance(value, dict) and "locator" in value else ""), True)
    own = materials.get("own_sealed_facts")
    fact_match_coverage = None
    if own is not None:
        fact_ids = [row["fact_id"] for row in own["data"]["findings"]["necessary_facts"]]
        matched_rows = findings.get("fact_matches", [])
        matched_ids = [row.get("sealed_fact_id") for row in matched_rows if isinstance(row, dict)] if isinstance(matched_rows, list) else []
        fact_match_coverage = len(fact_ids) == len(set(fact_ids)) and len(matched_ids) == len(fact_ids) and set(matched_ids) == set(fact_ids)
        if not fact_match_coverage:
            errors.append("own_sealed_fact_match_records_missing_duplicate_or_unknown")
    return {"stage": stage, "mechanical_valid": not errors, "declared_complete": answer.get("complete"),
        "review_complete": not errors and not unreviewed and not other_unreviewed and answer.get("complete") is True,
        "inventory_items": len(inventory_ids), "inventory_items_recorded": len(seen),
        "unreviewed_inventory_ids": unreviewed, "missing_inventory_ids": missing,
        "other_unreviewed_count": len(other_unreviewed), "own_fact_match_id_coverage": fact_match_coverage,
        "references_checked": len(references), "errors": errors,
        "semantic_support_proven": False, "atomic_claim_completeness_proven": False,
        **({"resolved_references": resolved_references} if anchor_mode else {})}
