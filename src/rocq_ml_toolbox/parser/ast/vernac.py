from __future__ import annotations

from typing import Any, Optional, List

from .model import (
    VernacElement,
    VernacKind
)
from .span import extract_span
from .base import MalformedAST, as_list, ensure_str, find_local_attr, first, jmaybe, jpath


def keyword_of(obj: dict) -> str:
    return ensure_str(jpath(obj, "v", "expr", 1, 0))


def parse_require(obj: dict) -> VernacElement:
    span = extract_span(obj)
    path_ast = jmaybe(obj, "v", "expr", 1, 1, default=None)
    is_import_ast = jmaybe(obj, "v", "expr", 1, 2, default=None)
    libs_ast = jmaybe(obj, "v", "expr", 1, 3, default=[])

    origin = None
    if path_ast:
        origin = jmaybe(path_ast, "v", 2, 1, default=None)

    import_kw = False
    if is_import_ast:
        try:
            import_kw = bool(is_import_ast[0] and is_import_ast[0][0] == "Import")
        except Exception:
            import_kw = False

    libs: list[str] = []
    for entry in as_list(libs_ast):
        libs.append(ensure_str(jmaybe(entry, 0, "v", 2, 1, default=entry)))

    return VernacElement(
        span=span,
        kind=VernacKind.REQUIRE,
        data={"libs": libs, "import": import_kw, "origin": origin},
    )


def parse_import(obj: dict) -> VernacElement:
    span = extract_span(obj)
    libs_ast = jmaybe(obj, "v", "expr", 1, 2, default=[])
    libs: list[str] = []
    for deps in as_list(libs_ast):
        libs.append(ensure_str(jmaybe(deps, 0, "v", 2, 1, default=deps)))
    return VernacElement(span=span, kind=VernacKind.IMPORT, data={"libs": libs})


def parse_open_close_scope(obj: dict) -> VernacElement:
    span = extract_span(obj)
    is_open = bool(jmaybe(obj, "v", "expr", 1, 1, default=False))
    name = ensure_str(jmaybe(obj, "v", "expr", 1, 2, default=None))
    attrs = jmaybe(obj, "v", "attrs", default=[])
    is_local = find_local_attr(attrs)
    return VernacElement(
        span=span,
        kind=VernacKind.OPEN_CLOSE_SCOPE,
        name=name,
        data={"open": is_open, "local": is_local},
    )


def parse_define_module(obj: dict) -> VernacElement:
    span = extract_span(obj)
    is_alias = jpath(obj, "v", "expr", 1, -1) != []
    name = jmaybe(obj, "v", "expr", 1, 2, "v", 1, default=None)
    return VernacElement(span=span, kind=VernacKind.DEFINE_MODULE, name=name, data={"is_alias": is_alias})

def parse_declare_module_type(obj: dict) -> VernacElement:
    span = extract_span(obj)
    is_alias = jpath(obj, "v", "expr", 1, -1) != []
    name = jmaybe(obj, "v", "expr", 1, 1, "v", 1, default=None)
    return VernacElement(span=span, kind=VernacKind.DECLARE_MODULE_TYPE, name=name, data={"is_alias": is_alias})

def parse_definition(obj: dict) -> VernacElement:
    span = extract_span(obj)
    subinfo = jmaybe(obj, "v", "expr", 1, 2, 0, "v", default=None)

    name: Optional[str] = None
    if isinstance(subinfo, list) and subinfo:
        if subinfo[0] == "Anonymous":
            name = None
        else:
            name = ensure_str(jmaybe(subinfo, 1, 1, default=None))
    return VernacElement(span=span, kind=VernacKind.DEFINITION, name=name)


def parse_start_theorem_proof(obj: dict) -> VernacElement:
    span = extract_span(obj)
    # prototype path preserved with safety
    name = ensure_str(jmaybe(obj, "v", "expr", 1, 2, 0, 0, 0, "v", 1, default=None))
    return VernacElement(span=span, kind=VernacKind.START_THEOREM_PROOF, name=name)


def parse_notation(obj: dict) -> VernacElement:
    span = extract_span(obj)
    name = ensure_str(jmaybe(obj, "v", "expr", 1, 2, "ntn_decl_string", "v", default=None))
    return VernacElement(span=span, kind=VernacKind.NOTATION, name=name)


def parse_reserved_notation(obj: dict) -> VernacElement:
    span = extract_span(obj)
    name = ensure_str(jmaybe(obj, "v", "expr", 1, 2, 0, "v", default=None))
    return VernacElement(span=span, kind=VernacKind.RESERVED_NOTATION, name=name)


def parse_fixpoint(obj: dict) -> VernacElement:
    span = extract_span(obj)
    name = ensure_str(jmaybe(obj, "v", "expr", 1, 2, 1, 0, "fname", "v", 1, default=None))
    return VernacElement(span=span, kind=VernacKind.FIXPOINT, name=name)


def parse_cofixpoint(obj: dict) -> VernacElement:
    span = extract_span(obj)
    name = ensure_str(jmaybe(obj, "v", "expr", 1, 2, 0, "fname", "v", 1, default=None))
    return VernacElement(span=span, kind=VernacKind.COFIXPOINT, name=name)


def parse_end_segment(obj: dict) -> VernacElement:
    span = extract_span(obj)
    name = ensure_str(jmaybe(obj, "v", "expr", 1, 1, "v", 1, default=None))
    return VernacElement(span=span, kind=VernacKind.END_SEGMENT, name=name)


def parse_begin_section(obj: dict) -> VernacElement:
    span = extract_span(obj)
    name = ensure_str(jmaybe(obj, "v", "expr", 1, 1, "v", 1, default=None))
    return VernacElement(span=span, kind=VernacKind.BEGIN_SECTION, name=name)


def parse_instance(obj: dict) -> VernacElement:
    span = extract_span(obj)
    subobj = jmaybe(obj, "v", "expr", 1, 1, 0, "v", default=None)
    name: Optional[str] = None
    if isinstance(subobj, list) and subobj:
        if subobj[0] == "Anonymous":
            name = None
        else:
            name = ensure_str(jmaybe(subobj, 1, 0, default=None))
    return VernacElement(span=span, kind=VernacKind.INSTANCE, name=name)


def parse_extend(obj: dict) -> Optional[VernacElement]:
    """
    Handle known 'VernacExtend' entries (ElpiHB*, VernacSolve -> proof step).
    Otherwise return None.
    """
    span = extract_span(obj)
    subinfo = jmaybe(obj, "v", "expr", 1, 1, default=None)
    if isinstance(subinfo, dict) and "ext_entry" in subinfo:
        name = ensure_str(subinfo["ext_entry"])
        if name.startswith("ElpiHB"):
            return VernacElement(span=span, kind=VernacKind.EXTEND, name=name, data={"family": "ElpiHB"})
        if name == "VernacSolve":
            return VernacElement(span=span, kind=VernacKind.PROOF_STEP, name=None)
        if name == 'VernacDeclareTacticDefinition':
            tac_name = jmaybe(obj, "v", "expr", 1, 2, 0, 2, 0, 1, 0, 1, 1, default=None)
            return VernacElement(span=span, kind=VernacKind.LTAC, name=tac_name)
    return None


def parse_inductive(obj: dict) -> VernacElement:
    """
    Parses a subset of 'VernacInductive'.
    """
    span = extract_span(obj)

    kind_raw = jmaybe(obj, "v", "expr", 1, 1, 0, default=None)
    kind_raw = ensure_str(kind_raw)

    name = ensure_str(jpath(obj, "v", "expr", 1, 2, 0, 0, 0, 1, 0, "v", 1))

    fields = jmaybe(obj, "v", "expr", 1, 2, 0, 0, default=None)
    if not isinstance(fields, list) or len(fields) < 4:
        raise MalformedAST("Unexpected VernacInductive shape (fields)")

    body = fields[3]
    if not isinstance(body, list) or not body:
        raise MalformedAST("Unexpected VernacInductive body")

    # Normalize kind
    def norm_kind() -> VernacKind:
        if kind_raw in ("Inductive_kw"):
            return VernacKind.INDUCTIVE
        if kind_raw == "CoInductive":
            return VernacKind.COINDUCTIVE
        if kind_raw == "Record":
            return VernacKind.RECORD
        if kind_raw == "Structure":
            return VernacKind.STRUCTURE
        if kind_raw == "Variant":
            return VernacKind.VARIANT
        if kind_raw == "Class":
            return VernacKind.CLASS
        raise MalformedAST(f"Unsupported inductive kind: {kind_raw}")

    kind = norm_kind()
    def collect_record_fields(record_fields_ast, kind: VernacKind) -> list[VernacElement]:
        out: list[VernacElement] = []
        for child in as_list(record_fields_ast):
            if jmaybe(child, 0, 1, "v", 0, default=None) == "Name":
                child_name = ensure_str(jmaybe(child, 0, 1, "v", 1, 1, default=""))
                member = VernacElement(span=extract_span(child), name=child_name, kind=kind)
                out.append(member)
        return out

    def collect_fields(ctors_ast, kind: VernacKind) -> list[VernacElement]:
        out: list[Member] = []
        for ctor in as_list(ctors_ast):
            ctor_name = ensure_str(jmaybe(ctor, 1, 0, "v", 1, default=None))
            member = Member(span=extract_span(ctor), name=ctor_name, kind=kind)
            out.append(member)
        return out

    def collect_constructor(ctor_ast) -> List[VernacElement]:
        constructor_name = ensure_str(jmaybe(ctor_ast, "v", 1, default=None))
        if constructor_name:
            constructor_span = extract_span(ctor_ast)
            return [VernacElement(name=constructor_name, span=constructor_span, kind=VernacKind.CONSTRUCTOR)]
        return []
    
    tag = ensure_str(body[0])

    if kind == VernacKind.VARIANT:
        members = collect_fields(jmaybe(body, 1, default=[]), kind=VernacKind.CONSTRUCTOR)
        return VernacElement(
            span=span,
            kind=kind,
            name=name,
            members=members
        )
    if kind == VernacKind.COINDUCTIVE or kind == VernacKind.INDUCTIVE:
        if tag == "RecordDecl":
            ctor_ast = jmaybe(body, 1, default=None)
            members = collect_constructor(ctor_ast)
            members.extend(collect_record_fields(jmaybe(body, 2, default=[]), kind=VernacKind.FIELD))
            return VernacElement(
                span=span,
                kind=kind,
                name=name,
                members=members
            )
        if tag == "Constructors":
            members = collect_fields(jmaybe(body, 1, default=[]), kind=VernacKind.CONSTRUCTOR)
            return VernacElement(
                span=span,
                kind=kind,
                name=name,
                members=members
            )
    if kind == VernacKind.STRUCTURE or kind == VernacKind.RECORD:
        ctor_ast = jmaybe(body, 1, default=None)
        members = collect_constructor(ctor_ast)
        members.extend(collect_record_fields(jmaybe(body, 2, default=[]), kind=VernacKind.FIELD))
        return VernacElement(
            span=span,
            kind=kind,
            name=name,
            members=members
        )
    if kind == VernacKind.CLASS:
        ctor_ast = jmaybe(body, 1, default=None)
        members = []
        if isinstance(ctor_ast, list):
            members = collect_fields(ctor_ast, kind=VernacKind.CONSTRUCTOR)
            return VernacElement(
                span=span,
                kind=kind,
                name=name,
                members=members
            )
        else:
            members = collect_constructor(ctor_ast)
        if 2 < len(body):
            members.extend(collect_record_fields(jmaybe(body, 2, default=[]), kind=VernacKind.FIELD))
            return VernacElement(
                span=span,
                kind=kind,
                name=name,
                members=members
            )
    raise MalformedAST(f"Unsupported VernacInductive body tag: {tag} (kind={kind_raw})")


def simple(kind: VernacKind):
    def _p(obj: dict) -> VernacElement:
        return VernacElement(span=extract_span(obj), kind=kind)

    return _p
