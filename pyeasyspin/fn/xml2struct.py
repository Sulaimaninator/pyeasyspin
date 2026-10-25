"""Convert XML file into a MATLAB structure"""
import os
import re
import xml.dom.minidom as _md
from .._struct import Struct


def _fixname(n):
    return n.replace('-', '_dash_').replace(':', '_colon_').replace('.', '_dot_')


def xml2struct(fileName=None):
    """Struct tree of an XML file (xml2struct by W. Falkena, as shipped with EasySpin):
    elements -> fields (repeated names -> list), attributes -> field 'Attributes', text ->
    field 'Text'. Accepts a file name (adds '.xml' if needed) or a minidom Document/Element."""
    if fileName is None:
        raise ValueError('xml2struct: no file name given.')
    if isinstance(fileName, _md.Node):
        xDoc = fileName
    else:
        if not os.path.exists(fileName):
            if not fileName[-4:] == '.xml':
                fileName = fileName + '.xml'
            if not os.path.exists(fileName):
                raise ValueError('The file ' + fileName + ' could not be found.')
        xDoc = _md.parse(fileName)
        xDoc.normalize()   # one text node per run of text, as Xerces delivers it
    return _parseChildNodes(xDoc)[0]


def _text_content(node):
    """DOM getTextContent (text and CDATA of all descendants; comments/PIs excluded)."""
    if node.nodeType in (node.TEXT_NODE, node.CDATA_SECTION_NODE, node.COMMENT_NODE,
                         node.PROCESSING_INSTRUCTION_NODE):
        return node.data
    return ''.join(_text_content(c) for c in node.childNodes
                   if c.nodeType not in (c.COMMENT_NODE, c.PROCESSING_INSTRUCTION_NODE))


def _parseChildNodes(theNode):
    children = Struct()
    ptext = Struct()
    textflag = 'Text'
    for theChild in theNode.childNodes:
        text, name, attr, childs, textflag = _getNodeData(theChild)
        if name not in ('#text', '#comment', '#cdata_dash_section'):
            if name in children:
                if not isinstance(children[name], list):
                    children[name] = [children[name]]
                item = childs
                if len(text) > 0:
                    item = text
                if attr is not None:
                    item['Attributes'] = attr
                children[name].append(item)
            else:
                item = childs
                if len(text) > 0:
                    item = text
                if attr is not None:
                    item['Attributes'] = attr
                children[name] = item
        else:
            ptextflag = 'Text'
            if name == '#cdata_dash_section':
                ptextflag = 'CDATA'
            elif name == '#comment':
                ptextflag = 'Comment'
            t = text[textflag]
            if re.sub(r'\s*', '', t):
                if ptextflag not in ptext or not ptext[ptextflag]:
                    ptext[ptextflag] = t
                else:
                    ptext[ptextflag] = ptext[ptextflag] + t
    return children, ptext, textflag


def _getNodeData(theNode):
    name = _fixname(theNode.nodeName)
    attr = _parseAttributes(theNode)
    if len(attr) == 0:
        attr = None
    childs, text, textflag = _parseChildNodes(theNode)
    if len(childs) == 0 and len(text) == 0:
        text = Struct()
        text[textflag] = _text_content(theNode)
    return text, name, attr, childs, textflag


def _parseAttributes(theNode):
    attributes = Struct()
    a = getattr(theNode, 'attributes', None)
    if a:
        # Xerces keeps attributes sorted by name
        for k in sorted(a.keys()):
            attributes[_fixname(k)] = a[k].value
    return attributes
