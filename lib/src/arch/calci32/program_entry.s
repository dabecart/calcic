    # Set the reset vector.
    .section reset_v
    .long _start

    .section text
    .globl	_start
_start:
    # Initialize the stack.
    mov     $__stack_start, %rsp
    mov     %rsp, %rsb

    # Call the main function.
    fun     main

    # Infinite loop.
_end:
    brk
    jmp     _end
