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
import enum
import sys


from . import lexer, parser, TAC
from . import TAC_optimizer as optimizer
from .builtin.builtin_functions import BuiltInFunctions
from .global_context import globalContext

from .x64 import builtin_types_x64
from .x64 import assembler_x64

class TargetArchitectures(enum.Enum):
    x64 = "x64"

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
                            description='A basic C99 compiler.',
                            epilog='Written by @dabecart, 2025.')
    argParser.add_argument('input_file',
                           type=str)
    argParser.add_argument('-o',
                           help="Place the output into <OUTPUT>.",
                           metavar="OUTPUT",
                           dest="output")
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
    argParser.add_argument("-O", "--optimize",
                           help="Enables all optimizations. You can specify the number of iterations of the optimization algorithm.",
                           type=int, 
                           choices=range(1, optimizer.MAX_ITERATION_STEPS + 1), 
                           metavar="ITERS",
                           nargs='?', const=optimizer.MAX_ITERATION_STEPS, default=0)
    argParser.add_argument("-march",
                           help="Select the target architecture.",
                           dest="architecture",
                           choices=[e.value for e in TargetArchitectures],
                           default=TargetArchitectures.x64.value)

    # Test options.
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
    
    args = argParser.parse_args(splitCombinedArguments(sys.argv[1:], {'l'}))
    inputFile: str = args.input_file
    inputFileBasename: str = inputFile.rsplit(".", 1)[0]

    """
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    PREPROCESSOR
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    """
    preprocessStatus = subprocess.run(["gcc", "-E", inputFile, "-o", f"{inputFileBasename}.i"])
    retCode = preprocessStatus.returncode

    if retCode != 0:
        exit(retCode)

    """
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    LEXER
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    """
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

    # The global context stores variables used on all stages of the compiler.
    # Add the built-in function handlers to the global context.
    BuiltInFunctions.connectHandlersToContext(globalContext)

    """
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    PARSER
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    """
    try:
        # Create the context.
        context = parser.Context()
        
        # Add built-in types to the context before parsing.
        match TargetArchitectures(args.architecture):
            case TargetArchitectures.x64:
                builtin_types_x64.BuiltInTypes_x64(context)
        
        # Parse the program.
        program = parser.Program(tokens, context)
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

    """
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    Three Address Code (TAC)
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    """
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

    """
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    OPTIMIZER
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    """
    if (args.optimize == 0) and (
        args.fold_constants or args.eliminate_unreachable_code or \
        args.propagate_copies or args.eliminate_dead_stores):
        # For the case when only the flags are used, but not the -O flag.
        iterationSteps = optimizer.MAX_ITERATION_STEPS
    else:
        iterationSteps = args.optimize

    optimizationFlags = optimizer.TACOptimizationFlags(
        constant_folding                = args.fold_constants               or bool(args.optimize),
        unreachable_code_elimination    = args.eliminate_unreachable_code   or bool(args.optimize),
        copy_propagation                = args.propagate_copies             or bool(args.optimize),
        dead_store_elimination          = args.eliminate_dead_stores        or bool(args.optimize),
        iteration_steps                 = iterationSteps
    )

    try:
        tacOptimizer = optimizer.TACOptimizer(optimizationFlags)
        # Modifies the inner instructions of the program.
        tacOptimizer.optimize(tacProgram)
        if args.verbose:
            print(f"Optimizer:\n{tacProgram}")
    except Exception as e:
        print(f"Optimizer exception:\n{e}", file=sys.stderr)
        if args.verbose:
            print(traceback.format_exc(), file=sys.stderr)
        exit(1)

    """
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ASSEMBLY GENERATION
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    """
    try:
        match TargetArchitectures(args.architecture):
            case TargetArchitectures.x64:
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

    """
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ASSEMBLER AND LINKER
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    """
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
    