"""
calcic.py

Entry point of the compiler. Handles command arguments, input routes and subcalls to the 
preprocessor and linker.

calcic. Written by @dabecart, 2026.
"""

import argparse
import subprocess
import os
import traceback

from . import assembler_x64, lexer, parser, TAC

import sys

def splitCombinedArguments(argv, initialValues: set):
    processedArgs = []
    for arg in argv:
        if arg.startswith('-') and not arg.startswith('--') and len(arg) > 2:
            opt = arg[1]
            rest = arg[2:]
            if opt in initialValues:
                processedArgs.append('-' + opt)
                processedArgs.append(rest)
            else:
                # Leave bundled flags like -abc as they are
                for ch in arg[1:]:
                    processedArgs.append('-' + ch)
        else:
            processedArgs.append(arg)
    return processedArgs

def main() -> None:
    argParser = argparse.ArgumentParser(
                            prog='calcic',
                            description='C compiler for the CALCI-16 chip.',
                            epilog='Written by @dabecart, 2025.')
    argParser.add_argument('input_file',
                           type=str)
    argParser.add_argument('-o',
                           help="Place the output into <OUTPUT>.",
                           metavar="OUTPUT",
                           dest="output")
    argParser.add_argument('--lex',
                           help="Only runs the lexer.",
                           action="store_true")
    argParser.add_argument('--parse',
                           help="Runs the lexer and then the parser.",
                           action="store_true")
    argParser.add_argument('--validate',
                           help="Runs the lexer and then the parser.",
                           action="store_true")
    argParser.add_argument('--tac',
                           help="Runs the lexer and the parser and converts to Three Address Code (TAC).",
                           action="store_true")
    argParser.add_argument("--codegen",
                           help="Performs lexing, parsing and assembly generation.",
                           action="store_true")
    argParser.add_argument("--fold-constants",
                           help="Enables constant folding.",
                           action="store_true")
    argParser.add_argument("--propagate-copies",
                           help="Enables copy propagation.",
                           action="store_true")
    argParser.add_argument("--eliminate-unreachable-code",
                           help="Enables unreachable code elimination.",
                           action="store_true")
    argParser.add_argument("--eliminate-dead-stores",
                           help="Enables dead store elimination.",
                           action="store_true")
    argParser.add_argument("-O", "--optimize",
                           help="Enables all optimizations.",
                           action="store_true")
    argParser.add_argument("-c",
                           help="Compile and assemble, but do not link. Generates a .o file.",
                           action="store_true",
                           dest="generate_object")
    argParser.add_argument("-S",
                           help="Compile but don't assemble. Generates a .s file.",
                           action="store_true",
                           dest="generate_assembly")
    argParser.add_argument("-l",
                           type=str,
                           help="Link a library.",
                           dest="library")
    argParser.add_argument("-v", "--verbose",
                           help="Prints insightful information.",
                           action="store_true")
    
    args = argParser.parse_args(splitCombinedArguments(sys.argv[1:], {'l'}))
    inputFile: str = args.input_file
    inputFileBasename: str = inputFile.rsplit(".", 1)[0]

    # Preprocessing.
    preprocessStatus = subprocess.run(["gcc", "-E", inputFile, "-o", f"{inputFileBasename}.i"])
    retCode = preprocessStatus.returncode

    if retCode != 0:
        exit(retCode)

    # Compilation.
    try:
        tokens = lexer.lex(f"{inputFileBasename}.i")
        if args.verbose:
            print(f"Tokens:\n{tokens}\n")
    except Exception as e:
        os.remove(f"{inputFileBasename}.i")
        print(f"Lexer exception:\n{e}", file=sys.stderr)
        if args.verbose:
            print(traceback.format_exc(), file=sys.stderr)
        exit(1)

    os.remove(f"{inputFileBasename}.i")

    if args.lex:
        exit(0)

    try:
        program = parser.Program(tokens)
        if len(tokens) > 0:
            raise ValueError("Missing tokens out of the program")
        
        if args.verbose:
            print(f"Parser:\n{program}")
    except Exception as e:
        print(f"Parser exception:\n{e}", file=sys.stderr)
        if args.verbose:
            print(traceback.format_exc(), file=sys.stderr)
        exit(1)

    if args.parse or args.validate:
        exit(0)

    try:
        tacProgram = TAC.TACProgram(program)
        if args.verbose:
            print(f"TAC:\n{tacProgram}")
    except Exception as e:
        print(f"TAC exception:\n{e}", file=sys.stderr)
        if args.verbose:
            print(traceback.format_exc(), file=sys.stderr)
        exit(1)

    if args.tac:
        exit(0)

    try:
        assemblyProgram = assembler_x64.AssemblerProgram(tacProgram)
        # if args.verbose:
        #     print(f"Assembler:\n{assemblyProgram}")
    except Exception as e:
        print(f"Assembler exception:\n{e}", file=sys.stderr)
        if args.verbose:
            print(traceback.format_exc(), file=sys.stderr)
        exit(1)

    if args.codegen:
        exit(0)

    assemblyCode = assemblyProgram.emitCode()
    if args.verbose:
        print(f"Assembly code:\n{assemblyCode}")

    if args.generate_assembly and args.output is not None:
        assemblyOutput = args.output
    else:
        assemblyOutput = f"{inputFileBasename}.s"

    with open(assemblyOutput, "w") as assemblyFile:
        assemblyFile.write(assemblyCode)

    if args.generate_assembly:
        exit(0)

    # Assembly and link.
    if args.output is None:
        if args.generate_object:
            exeOutput = f"{inputFileBasename}.o"
        else:
            exeOutput = inputFileBasename
    else:
        exeOutput = args.output

    assemblyCommand: list[str] = ["gcc", f"{inputFileBasename}.s", "-o", exeOutput]
    if args.generate_object:
        assemblyCommand.append("-c")
    if args.library:
        for lib in args.library:
            assemblyCommand.append("-l")
            assemblyCommand.append(lib)
            
    assemblyStatus = subprocess.run(assemblyCommand)
    retCode = assemblyStatus.returncode
    if retCode != 0:
        exit(retCode)

    os.remove(f"{inputFileBasename}.s")
    