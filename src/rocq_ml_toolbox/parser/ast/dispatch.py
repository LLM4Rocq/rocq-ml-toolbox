from __future__ import annotations

from typing import Callable, Optional

from .model import AstNode, UnsupportedNode, VernacElement, VernacKind
from .span import extract_span
from .vernac import (
    keyword_of,
    parse_begin_section,
    parse_cofixpoint,
    parse_define_module,
    parse_definition,
    parse_end_segment,
    parse_extend,
    parse_fixpoint,
    parse_import,
    parse_inductive,
    parse_instance,
    parse_notation,
    parse_open_close_scope,
    parse_require,
    parse_reserved_notation,
    parse_start_theorem_proof,
    parse_declare_module_type,
    parse_syntactic_definition,
    simple,
)

Parser = Callable[[dict], AstNode]


PARSERS: dict[str, Callable[[dict], AstNode | None]] = {
    "VernacRequire": parse_require,
    "VernacImport": parse_import,
    "VernacOpenCloseScope": parse_open_close_scope,
    "VernacSetOption": simple(VernacKind.SET_OPTION),
    "VernacDefineModule": parse_define_module,
    "VernacExtend": parse_extend,  # may return None
    "VernacBindScope": simple(VernacKind.BIND_SCOPE),
    "VernacEndSegment": parse_end_segment,
    "VernacBeginSection": parse_begin_section,
    "VernacContext": simple(VernacKind.CONTEXT),
    "VernacDefinition": parse_definition,
    "VernacArguments": simple(VernacKind.ARGUMENTS),
    "VernacSyntacticDefinition": parse_syntactic_definition,
    "VernacReserve": simple(VernacKind.RESERVE),
    "VernacStartTheoremProof": parse_start_theorem_proof,
    "VernacProof": simple(VernacKind.PROOF),
    "VernacEndProof": simple(VernacKind.END_PROOF),
    "VernacAbort": simple(VernacKind.ABORT),
    "VernacAssumption": simple(VernacKind.ASSUMPTION),
    "VernacHints": simple(VernacKind.HINTS),
    "VernacBullet": simple(VernacKind.BULLET),
    "VernacNotation": parse_notation,
    "VernacReservedNotation": parse_reserved_notation,
    "VernacCoFixpoint": parse_cofixpoint,
    "VernacFixpoint": parse_fixpoint,
    "VernacDeclareScope": simple(VernacKind.DECLARE_SCOPE),
    "VernacDelimiters": simple(VernacKind.DELIMITERS),
    "VernacInductive": parse_inductive,
    "VernacCoercion": simple(VernacKind.COERCION),
    "VernacCanonical": simple(VernacKind.CANONICAL),
    "VernacInstance": parse_instance,
    "VernacInclude": simple(VernacKind.INCLUDE),
    "VernacDeclareCustomEntry": simple(VernacKind.DECLARE_CUSTOM_ENTRY),
    "VernacDeclareModuleType": parse_declare_module_type,
    "VernacDeclareModule": simple(VernacKind.DECLARE_MODULE),
    "VernacIdentityCoercion": simple(VernacKind.IDENTITY_COERCION),
    "VernacAttributes": simple(VernacKind.ATTRIBUTES),
    "VernacRemoveHints": simple(VernacKind.REMOVE_HINTS),
    "VernacSubproof": simple(VernacKind.SUBPROOF),
    "VernacEndSubproof": simple(VernacKind.END_SUBPROOF),
    "VernacDeclareInstance": simple(VernacKind.DECLARE_INSTANCE),
    "VernacCreateHintDb": simple(VernacKind.CREATE_HINT_DB),
    "VernacDeclareMLModule": simple(VernacKind.DECLARE_ML_MODULE),
    "VernacExistingInstance": simple(VernacKind.EXISTING_INSTANCE),
    "VernacScheme": simple(VernacKind.SCHEME),
    "VernacSetOpacity": simple(VernacKind.SET_OPACITY),
    "VernacExactProof": simple(VernacKind.EXACT_PROOF),
    "VernacRegister": simple(VernacKind.REGISTER),
    "VernacSetStrategy": simple(VernacKind.SET_STRATEGY),
    "VernacAddOption": simple(VernacKind.ADD_OPTION),
    "VernacGeneralizable": simple(VernacKind.GENERALIZABLE),
    "VernacExtraDependency": simple(VernacKind.EXTRA_DEPENDENCY),
}


def parse_node(
    obj: dict,
    *,
    on_unsupported: str = "keep",  # "keep" | "raise"
    keep_raw: bool = False,
) -> VernacElement:
    kw = keyword_of(obj)
    parser = PARSERS.get(kw)

    if parser is None:
        if on_unsupported == "raise":
            raise NotImplementedError(f"Unsupported vernac keyword: {kw}")
        return UnsupportedNode(kind=VernacKind.UNKNOWN ,span=extract_span(obj), keyword=kw, raw=obj if keep_raw else {"keyword": kw})

    out = parser(obj)
    if out is None:
        # e.g. VernacExtend unknown entry
        if on_unsupported == "raise":
            raise NotImplementedError(f"Parser returned None for: {kw}")
        return VernacElement(span=extract_span(obj), kind=VernacKind.UNKNOWN, name=kw)

    return out
