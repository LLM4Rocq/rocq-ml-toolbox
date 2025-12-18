from enum import StrEnum

from pytanque.protocol import TocElement

from ..parser import Element

class NotationDetail(StrEnum):
    NOTATION = "Notation"
    TACTIC_NOTATION = "Tactic Notation"

class TacticDetail(StrEnum):
    TACTIC = "Tactic"

class InductiveDetail(StrEnum):
    INDUCTIVE = "Inductive"
    COINDUCTIVE = "CoInductive"
    VARIANT = "Variant"
    RECORD = "Record"
    STRUCTURE = "Structure"
    CLASS = "Class"
    CONSTRUCTOR = "Constructor"
    FIELD = "Field"

class AssumptionDetail(StrEnum):
    VARIABLE = "Variable"
    AXIOM = "Axiom"
    PARAMETER = "Parameter"
    CONTEXT = "Context"

class DefinitionDetail(StrEnum):
    DEFINITION = "Definition"
    COERCION = "Coercion"
    SUBCLASS = "SubClass"
    CANONICAL_STRUCTURE = "CanonicalStructure"
    EXAMPLE = "Example"
    FIXPOINT = "Fixpoint"
    COFIXPOINT = "CoFixpoint"
    SCHEME = "Scheme"
    STRUCTURE_COMPONENT = "StructureComponent"
    IDENTITY_COERCION = "IdentityCoercion"
    INSTANCE = "Instance"
    METHOD = "Method"
    LET = "Let"
    LET_CONTEXT = "LetContext"

class TheoremDetail(StrEnum):
    THEOREM = "Theorem"
    LEMMA = "Lemma"
    FACT = "Fact"
    REMARK = "Remark"
    PROPERTY = "Property"
    PROPOSITION = "Proposition"
    COROLLARY = "Corollary"

ALL_DETAILS = [TheoremDetail, DefinitionDetail, AssumptionDetail, InductiveDetail, NotationDetail, TacticDetail]

def merge_toc_element(element: Element, toc_element:TocElement):
    name = toc_element.name.v
    range = toc_element.range
    element = Element(name)