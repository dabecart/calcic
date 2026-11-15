    # ----------------------------------------
    # Boot up vectors.
    # ----------------------------------------
    # Set the IRQ vector.
    .section irq_v
    .long _irqh

    # Set the reset vector.
    .section reset_v
    .long _start

    # ----------------------------------------
    # IRQ Handler.
    # ----------------------------------------
    .section text
    .globl _irqh
_irqh:
    # Push the general registers.
    psh %r0
    psh %r1
    psh %r2
    psh %r3
    psh %r4
    psh %r5
    psh %r6
    psh %r7
    psh %r8
    psh %r9
    psh %r10
    psh %r11

    # Fetch the IRQ destination address and run the function.
    mov 0x7f000000, %r0
    fun %r0

    # Restore the general registers.
    pop %r11
    pop %r10
    pop %r9
    pop %r8
    pop %r7
    pop %r6
    pop %r5
    pop %r4
    pop %r3
    pop %r2
    pop %r1
    pop %r0

    # Exit IRQ handler.
    reti

    # ----------------------------------------
    # Program entry.
    # ----------------------------------------
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
