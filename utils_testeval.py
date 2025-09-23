import ast
import textwrap
import time
import os
import random
random.seed(42)
import signal
import subprocess
import json
from copy import deepcopy
import csv
from contextlib import redirect_stdout, redirect_stderr


def change_function_name(code, new_name):
    try:
        # Parse the code into an AST
        tree = ast.parse(code)

        # Find the first function definition and change its name
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name != new_name:
                    node.name = new_name
                    break
                else:
                    break

        # Convert the modified AST back to code
        new_code = ast.unparse(tree)
        return new_code
    except Exception as e:  #cannoot parse
        return code


def change_assertion(testcase):
    lines = testcase.split('\n')
    lineno = -1
    for i, line in enumerate(lines):
        if "assert" in line.lower():
            lineno = i
    if lineno != -1:
        if "." in lines[lineno].split("(")[0]:
            remove_text = lines[lineno].split(".")[0].strip() + "."
            new_text = lines[lineno].replace(remove_text, "")
            lines = lines[:lineno] + [new_text] + lines[lineno+1:]
    return '\n'.join(lines)


def reformat_case_byrules(testcase, func_name, num=0, lang='python'):
    if testcase.startswith(' '):  # remove extra indents (encountered in codellama, mistral-7b starts with one space...)
        testcase = textwrap.dedent(testcase)
    lines = testcase.split('\n')

    if lang == 'python':
        last_line = lines[-1]  # if last line is not complete (due to token limit), remove it    
        last_line = textwrap.dedent(last_line)
        try:
            compile(last_line, '<string>', 'exec')
        except:
            # print('imcomplete last line, remove it', last_line)
            lines = lines[:-1]  # last line cannot compile
    elif lang == 'java':
        first_line = lines[0].replace("SolutionTest", f"Test{num}")
        lines = [first_line] + lines[1:]
    testcase = '\n'.join(lines)
    if lang == 'python':
        testcase = change_function_name(testcase, func_name)
    elif lang == 'java':
        testcase = change_assertion(testcase)
    return testcase


def remove_extra(testcase, lang='python'):
    """Remove extra test inputs and natural language descriptions before and after the test method.
    Only keep the contents between def test() and solution.{func_name}"""
    lines = testcase.split('\n')
    func_startline = 0
    start_flag = {
        "python": "def test",
        "java": "class SolutionTest",
        "cpp": "int main"
    }[lang]
    for i in range(len(lines)):
        if lines[i].find(start_flag) >= 0:
            func_startline = i
            break
    test_endline = len(lines)
    for i in range(len(lines)):
        if (lang == "python" and lines[i].find(f'assert ') >= 0) or \
            (lang in ["java", "cpp"] and lines[i] == "}"):
            test_endline = i + 1
            break
    new_testcase = '\n'.join(lines[func_startline: test_endline])
    return new_testcase


class TimeoutHandler:
    def __init__(self, timeout, error_message=None):
        self.timeout = timeout
        self.error_message = error_message
    
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.raise_timeout) #SIGALRM only support unix
        signal.alarm(self.timeout)
    
    def __exit__(self, type, value, traceback):
        signal.alarm(0)
    
    def raise_timeout(self, *args):
        raise TimeoutError(self.error_message)


def execute(test_code,timeout=5):
    """try to execute test code"""
    try:
        exec_globals = {}
        with TimeoutHandler(timeout):
            with open(os.devnull, 'w') as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    exec(test_code, globals())
            return True
    except AssertionError: #assertionerror is considered as executable
        return True
    except TimeoutError:
        #print("timed out")
        return False
    except Exception as e:
        #print(f"failed: {type(e).__name__}")
        return type(e).__name__, e #return error type and error message


def concatenate_cpp(passed_tests, imports):
    final_code = imports
    for n in passed_tests:
        with open(f"test{n}.cpp", "r") as f:
            code = f.read()
        start_line = 0
        code_lines = code.splitlines()
        extracted_lines = []
        for i, code_line in enumerate(code_lines):
            if code_line.startswith("int main"):
                start_line = i + 1 if "{" in code_line else i + 2
                break
        for i in range(start_line, len(code_lines)):
            line = code_lines[i]
            if line.strip().lower().startswith("assert"):
                pos = line.lower().find("assert")
                extracted_lines.append(line[:pos] + line[pos + len("assert"):])
                break
            else:
                extracted_lines.append(line)
        while len(extracted_lines) > 0 and (extracted_lines[-1].strip() in ["}", ""] or extracted_lines[-1].strip().startswith("return")):
            extracted_lines.pop()
        final_code += f"void test{n}() " + "{\n" + "\n".join(extracted_lines) + "\n}\n\n"
    final_code += "int main() {\n"
    for n in passed_tests:
        final_code += f"    test{n}();\n"
    return final_code + "    return 0;\n}"


def process_sample(data, language):
    total_cases = 0
    total_syn_correct = 0
    total_exec_correct = 0

    syn_fails = []
    exec_fails = []

    total_line_cov = 0
    total_branch_cov = 0

    task_num = data['task_num']
    difficulty = data['difficulty']
    func_name = data['func_name']
    code = data['code']
    test_cases = data['tests']
    tmp_dir = f'tmp/tmp_{task_num}'

    os.makedirs(tmp_dir, exist_ok=True)  # create different tmp folders for different problems to avoid conflicts

    if language == "python":
        test_import = f'from tmp.tmp_{task_num}.under_test import Solution\n'
        test_import_simple = f'from under_test import Solution\n'
        with open(f'{tmp_dir}/under_test.py', 'w') as f:
            f.write(code)

        passed_tests = []
        for j, testcase in enumerate(test_cases):
            total_cases += 1
            try:
                res = compile(testcase, '<string>', 'exec') # check syntax correctness
                total_syn_correct += 1

                test_code = test_import + testcase + f'\ntest_{func_name}()'
                time.sleep(0.01)
                res = execute(test_code)
                if res == True:
                    if test_code.find(f'solution.{func_name}') == -1:  # if the function under test is not called, also consider as failed
                        # print(f'func {func_name} under test not called: {task_num}, {j}')
                        exec_fails.append({'task': task_num, 'test_num': j, 'error': 'not called'})
                    else:
                        total_exec_correct += 1
                        test_code_simple = test_import_simple + testcase  # write to files for computing coverage
                        with open(f'{tmp_dir}/test_{j}.py', 'w') as f:
                            f.write(test_code_simple)
                        passed_tests.append(f'test_{j}.py')
                else:
                    # print(f'failed to execute: {task_num}, {j}, {res}')
                    exec_fails.append({'task': task_num, 'test_num': j, 'error': res})
            except:
                syn_fails.append({'task': task_num, 'test_num': j, 'error': 'syntax error'})
                pass

        if len(passed_tests) > 0:
            cov_command_prefix = ['pytest', '--cov=under_test', '--cov-branch', '--cov-report=json:coverage.json']
            subprocess.run(f'cp .coveragerc {tmp_dir}/.coveragerc', shell=True)  # copy config file to tmp_folder
            os.chdir(tmp_dir)  # enter tmp_ folder for testing
            cov_command = deepcopy(cov_command_prefix)
            for test in passed_tests:
                cov_command.append(test)

            try:
                subprocess.run(cov_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                cov_report = json.load(open('coverage.json'))
                total_stmt = cov_report['totals']['num_statements']
                covered_stmt = cov_report['totals']['covered_lines']
                line_cov = covered_stmt / total_stmt
                total_branch = cov_report['totals']['num_branches']
                covered_branch = cov_report['totals']['covered_branches']
                branch_cov = covered_branch / total_branch
                total_line_cov += line_cov
                total_branch_cov += branch_cov
            except: # unknown pytest error: cannot generate coverage report (AssertionError: Expected current collector to be <Collector at 0x7f7d2db07810: CTracer>, but it's <Collector at 0x7f7d2cd794d0: CTracer>)
                # print('Failed to generate coverage report')
                pass

            os.chdir('../..')  # return to the original directory

        else:  # no test cases passed
            pass


    elif language == "java":
        main_imports = "import java.util.*;\nimport java.util.stream.*;\nimport javafx.util.Pair;\n\n"
        subprocess.run(f'cp pom.xml {tmp_dir}/pom.xml', shell=True)  # copy config file to tmp_folder
        cmd = f'CP=$(mvn dependency:build-classpath -Dmdep.outputFile=/dev/stdout -q -DincludeScope=test):target/classes; echo $CP'
        cp_value = subprocess.check_output(cmd, shell=True, text=True).strip()
        os.chdir(tmp_dir)
        os.makedirs(f"src/main/java", exist_ok=True)
        os.makedirs(f"src/test/java", exist_ok=True)
        with open(f"src/main/java/Solution.java", "w") as f:
            f.write(main_imports + code)
        cp_javafx_value = ":".join([file for file in cp_value.split(":") if "javafx" in file])
        subprocess.run(["javac", "-cp", cp_javafx_value, "-d", "target/classes", "src/main/java/Solution.java"], timeout=10)
        passed_tests = []
        testcase_imports = "import static org.junit.jupiter.api.Assertions.*;\nimport org.junit.jupiter.api.Test;\n\n"
        for j, testcase in enumerate(test_cases):
            total_cases += 1
            testcase_file = f"src/test/java/Test{j}.java"
            with open(testcase_file, "w") as f:
                f.write(testcase_imports + testcase)
            try:
                output = subprocess.run(
                    ["javac", "-cp", cp_value, "-d", "target/test-classes", testcase_file],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=10
                )
                if output.returncode == 0:
                    passed_tests.append(testcase_file)
                    total_exec_correct += 1
                else:
                    # print({'task': task_num, 'test_num': j, 'error': output.stderr})
                    exec_fails.append({'task': task_num, 'test_num': j, 'error': output.stderr})
                    os.rename(testcase_file, testcase_file.replace(".java", ".fail"))
            except Exception as e:
                # print({'task': task_num, 'test_num': j, 'error': str(e)})
                exec_fails.append({'task': task_num, 'test_num': j, 'error': str(e)})
                os.rename(testcase_file, testcase_file.replace(".java", ".fail"))
        if len(passed_tests) > 0:
            try:
                output = subprocess.run(
                    ["mvn", "clean", "test"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=20
                )
                with open("target/jacoco-ut/jacoco.csv", "r") as csvfile:
                    reader = csv.DictReader(csvfile)
                    result = reader.__next__()
                    missed_line = int(result["LINE_MISSED"])
                    covered_line = int(result["LINE_COVERED"])
                    missed_branch = int(result["BRANCH_MISSED"])
                    covered_branch = int(result["BRANCH_COVERED"])
                    line_cov = covered_line / (missed_line + covered_line)
                    branch_cov = covered_branch / (missed_branch + covered_branch)
                    total_line_cov += line_cov
                    total_branch_cov += branch_cov
            except Exception as e:
                # print({'task': task_num, 'error': f'Failed to generate coverage report: {e}'})
                pass
        os.chdir("../..")


    elif language == "cpp":
        solution_imports = "#include <bits/stdc++.h>\nusing namespace std;\n\n"
        os.chdir(tmp_dir)
        with open("solution.h", "w") as f:
            f.write(solution_imports + code)
        passed_tests = []
        testcase_imports = '#include <bits/stdc++.h>\n#include "solution.h"\nusing namespace std;\n\n'
        for j, testcase in enumerate(test_cases):
            total_cases += 1
            testcase_file = f"test{j}.cpp"
            with open(testcase_file, "w") as f:
                f.write(testcase_imports + testcase)
            try:
                output = subprocess.run(
                    ["g++", "-std=c++20", "-O0", f"test{j}.cpp", "-o", f"test{j}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=20
                )
                if output.returncode == 0:
                    passed_tests.append(j)
                    total_exec_correct += 1
                else:
                    # print({'task': task_num, 'test_num': j, 'error': output.stderr})
                    exec_fails.append({'task': task_num, 'test_num': j, 'error': output.stderr})
            except Exception as e:
                # print({'task': task_num, 'test_num': j, 'error': str(e)})
                exec_fails.append({'task': task_num, 'test_num': j, 'error': str(e)})
        if len(passed_tests) > 0:
            try:
                all_passed_tests = concatenate_cpp(passed_tests, testcase_imports)
                with open("main.cpp", "w") as f:
                    f.write(all_passed_tests)
                subprocess.run(
                    ["g++", "-std=c++20", "-O0", "--coverage", "main.cpp", "-o", "main"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=20
                )
                subprocess.run(["./main"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
                output = subprocess.run(
                    ["gcov", "-b", "-c", "-n", "main.cpp"],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=20
                )
                result = output.stdout
                lines = result.splitlines()
                for i, line in enumerate(lines):
                    if line.strip() == "File 'solution.h'":
                        line_cov = float(lines[i+1].split(":")[-1].split("%")[0]) / 100
                        branch_cov = float(lines[i+2].split(":")[-1].split("%")[0]) / 100
                        total_line_cov += line_cov
                        total_branch_cov += branch_cov
                        break
            except Exception as e:
                # print({'task': task_num, 'error': f'Failed to generate coverage report: {e}'})
                pass
        os.chdir("../..")

    else:
        raise ValueError(f"Unsupported language: {language}")

    return {
        'total_cases': total_cases,
        'syn_correct': total_syn_correct,
        'exec_correct': total_exec_correct,
        'syn_fails': syn_fails,
        'exec_fails': exec_fails,
        'total_line_cov': total_line_cov,
        'total_branch_cov': total_branch_cov,
    }

