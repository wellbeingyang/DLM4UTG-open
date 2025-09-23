from __future__ import annotations
import torch
from tree_sitter import Language, Parser, Node
import tree_sitter_python, tree_sitter_java, tree_sitter_cpp
from typing import Union, List
import numpy as np


class Line:
    def __init__(self, lineno, tokens, unmask_index):
        """
        unmask_index: 1 if the token is unmasked, 0 if masked
        """
        self.lineno = lineno
        assert len(tokens) == len(unmask_index)
        self.tokens = tokens
        self.unmask_index = unmask_index

    def __str__(self):
        return "".join(self.tokens)

    def dump(self):
        return self.__str__()

    def dump_code(self):
        return self.__str__().strip()

    def dump_with_color(self, color_id=34):
        result = ""
        current = 0
        for i, (token, unmasked) in enumerate(zip(self.tokens, self.unmask_index)):
            if current == 0 and unmasked:
                result += f"\033[{color_id}m"
            elif current == 1 and not unmasked:
                result += "\033[0m"
            token = token.rstrip("\n") if i == len(self.tokens) - 1 else token
            result += token
            if i == len(self.tokens) - 1 and unmasked:
                result += "\033[0m"
            current = unmasked
        return result

    def is_useless(self) -> bool:
        s = self.dump_code()
        return all(c.isspace() or c.isdigit() or c in special_chars for c in s)


def parse_into_lines(tokens, unmask_index):
    lines = []
    lineno = 0
    current_tokens = []
    current_unmask_index = []
    for i, (token, unmasked) in enumerate(zip(tokens, unmask_index)):
        current_tokens.append(token)
        current_unmask_index.append(unmasked)
        if token.endswith("\n") or i == len(tokens) - 1:
            lines.append(Line(lineno, current_tokens, current_unmask_index))
            lineno += 1
            current_tokens = []
            current_unmask_index = []
    return lines


language = None
parser = None
root_type = None
comment_type = None
int_type = None
float_type = None
none_type = None
true_type = "true"
false_type = "false"
excluded_types = None
special_chars = [",", "[", "]", "(", ")", "{", "}", "'", '"']
string_type = None
string_content_type = None
list_type = "list"
assert_type = "assert_statement"


def set_language(lang):
    global language, parser, root_type, comment_type, int_type, float_type, none_type, excluded_types
    language = lang
    if language == "python":
        parser = Parser(Language(tree_sitter_python.language()))
    elif language == "java":
        parser = Parser(Language(tree_sitter_java.language()))
    elif language == "cpp":
        parser = Parser(Language(tree_sitter_cpp.language()))
    else:
        raise ValueError(f"Unsupported language: {language}")
    root_type = {
        "python": "module",
        "java": "program",
        "cpp": "translation_unit"
    }[language]
    comment_type = {
        "python": "comment",
        "java": "line_comment",
        "cpp": "comment"
    }[language]
    int_type = {
        "python": "integer",
        "java": "int",
        "cpp": "number_literal"
    }[language]
    float_type = {
        "python": "float",
        "java": "decimal_floating_point_literal",
        "cpp": "number_literal"
    }[language]
    none_type = {
        "python": "none",
        "java": "null_literal",
        "cpp": "null"
    }[language]
    excluded_types = [root_type, comment_type, int_type, float_type, none_type, "true", "false"]

    global string_type, string_content_type
    string_type = {
        "python": "string",
        "java": "string_literal",
        "cpp": "string_literal"
    }
    string_content_type = {
        "python": "string_content",
        "java": "string_fragment",
        "cpp": "string_content"
    }


class MyNode():
    def __init__(self, node_or_type: Union[Node, MyNode, str], text: Union[bytes, None]=None):
        self.type = node_or_type
        self.text = text
        self.children: List[MyNode] = []
        if not isinstance(node_or_type, str):
            self.type = node_or_type.type
            self.text = node_or_type.text
            self.start_byte = node_or_type.start_byte
            self.end_byte = node_or_type.end_byte
            for child in node_or_type.children:
                self.children.append(MyNode(child))

    def __repr__(self):
        return f"MyNode(type={self.type})"

    def __str__(self, level=1):
        s = self.type
        for child in self.children:
            s += "\n" + "  " * level + child.__str__(level + 1)
        return s

    def dump(self, indent: bool=True):
        if indent:
            return self.__str__()
        else:
            if self.is_leaf:
                return self.type
            else:
                return self.type + "(" + " ".join(child.dump(indent=False) for child in self.children) + ")"

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def has_error(self):
        if self.type in ["ERROR", "MISSING"]:
            return True
        else:
            for child in self.children:
                if child.has_error:
                    return True
        return False

    def merge(self, other: Union[Node, MyNode]) -> MyNode:
        if self.type != other.type or self.type == "ERROR":
            return MyNode("placeholder")
        node = MyNode(self.type, self.text)
        for child, other_child in zip(self.children, other.children):
            if child.type != other_child.type or child.type == "ERROR":
                break
            elif child.type in excluded_types and child.type != root_type:
                node.children.append(MyNode("placeholder"))
            elif child.type == "identifier" and child.text != other_child.text:
                node.children.append(MyNode("placeholder"))
            else:
                merged_child = child.merge(other_child)
                node.children.append(merged_child)
        while len(node.children) > 0 and node.children[-1].type == "placeholder":
            node.children.pop()
        return node

    def common_positions(self, other: Node, mask: np.ndarray = None) -> np.ndarray:
        if mask is None:
            mask = np.zeros(other.end_byte, dtype=bool)
        if self.type == other.type:
            if other.child_count == 0:
                start, end = other.start_byte, other.end_byte
                mask[start:end] = (self.type not in excluded_types)
            else:
                for child, other_child in zip(self.children, other.children):
                    mask = child.common_positions(other_child, mask)
        return mask

    def is_empty(self) -> bool:
        if self.type not in excluded_types:
            return False
        for child in self.children:
            if not child.is_empty():
                return False
        return True


def find_code_lines(all_lines: List[Line]):
    code_lines = []
    find = False
    for line in all_lines:
        if not find:
            if f"```{language}" in line.dump_code():
                find = True
        else:
            if not torch.tensor(line.unmask_index, dtype=bool).all():
                code_lines.append(line.lineno)
                # try:
                #     ast.parse(line.dump_code())
                #     code_lines.append(line.lineno)
                # except:
                #     pass
            if "```" in line.dump_code() and f"```{language}" not in line.dump_code():
                find = False
    return code_lines


def tree_accelerator(x, unmask_index):
    assert len(x) == len(unmask_index)
    all_lines = parse_into_lines(x, unmask_index)
    code_lines = find_code_lines(all_lines)

    merged2lines = {}
    for lineno in code_lines:
        line = all_lines[lineno]
        success = False
        node = parser.parse(bytes(line.dump(), "utf-8")).root_node
        if not line.is_useless() and not MyNode(node).is_empty():
            for merged_node in list(merged2lines.keys()):
                new_merged_node = merged_node.merge(node)
                if not new_merged_node.is_empty():
                    merged2lines[new_merged_node] = merged2lines.pop(merged_node)
                    merged2lines[new_merged_node].append((line.lineno, node))
                    success = True
                    break
            if not success:
                merged2lines[MyNode(node)] = [(line.lineno, node)]

    for merged_node, lines in merged2lines.items():
        if len(lines) > 1 and not merged_node.is_empty():
            for lineno, node in lines:
                line = all_lines[lineno]
                unmask_positions = merged_node.common_positions(node)
                offset = 0
                for i, (token, unmasked) in enumerate(zip(line.tokens, line.unmask_index)):
                    if not unmasked:
                        current = unmask_positions[offset: offset + len(token)]
                        current |= np.char.isspace(np.array(list(token)))
                        if current.all().item() and token.strip() != "":
                            # print(token)
                            line.unmask_index[i] = True
                    offset += len(token)

    final_unmask_index = []
    for line in all_lines:
        final_unmask_index.extend(line.unmask_index)

    return torch.tensor(final_unmask_index, dtype=bool).to(unmask_index.device)
