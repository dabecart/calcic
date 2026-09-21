	.text
	.globl	_start
_start:
	pushq	%rbp
	movq	%rsp, %rbp
	subq	$16, %rsp
	movl	$0, %eax
	call	main@PLT
	movl	%eax, -4(%rbp)
	movl	-4(%rbp), %eax
	
    # Perform a syscall "exit" (code 60).
	movl %eax, %edi 
	movl $60, %eax
	syscall
	ret

	.section .note.GNU-stack,"",@progbits
