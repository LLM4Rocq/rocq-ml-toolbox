"""
Extract from CoqPyt library (https://github.com/sr-lab/coqpyt)
Highly restricted version, only to get the AST from rocq-lsp.
"""

import enum
from typing import List, Dict, Optional, Any

from pytanque.protocol import Range

class Diagnostic:
    def __init__(
        self,
        range,
        message,
        severity=None,
        code=None,
        codeDescription=None,
        source=None,
        tags=None,
        relatedInformation=None,
        data=None,
    ):
        """
        Constructs a new Diagnostic instance.
        """
        self.range: Range = Range(**range)
        self.severity = severity
        self.code = code
        self.source = source
        self.message = message
        self.relatedInformation = relatedInformation

class TextDocumentItem:
    """
    An item to transfer a text document from the client to the server.
    """

    def __init__(self, path:str, languageId="coq", version=1):
        """
        Constructs a new Diagnostic instance.
        """
        self.uri = f"file://{path}"
        with open(path, 'r') as file:
            self.text = file.read()
        self.languageId = languageId
        self.version = version

class TextDocumentIdentifier:
    """
    Text documents are identified using a URI. On the protocol level, URIs are passed as strings.
    """

    def __init__(self, uri):
        """
        Constructs a new TextDocumentIdentifier instance.
        """
        self.uri = uri


class ErrorCodes(enum.Enum):
    # Defined by JSON RPC
    ParseError = -32700
    InvalidRequest = -32600
    MethodNotFound = -32601
    InvalidParams = -32602
    InternalError = -32603
    serverErrorStart = -32099
    serverErrorEnd = -32000
    ServerTimeout = -32004
    ServerQuit = -32003
    ServerNotInitialized = -32002
    UnknownErrorCode = -32001

    # Defined by the protocol.
    RequestCancelled = -32800
    ContentModified = -32801


class ResponseError(Exception):
    def __init__(self, code, message, data=None):
        if isinstance(code, ErrorCodes):
            code = code.value
        self.code = code
        self.message = message
        if data:
            self.data = data
