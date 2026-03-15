# DWARF example

We'll use the following program as an example:

```c
int main() {
    int x = 0;
    int y = 1 + x;

    return x + y;
}
```

Use the following to compile:
```
gcc -O0 -gdwarf-${DWARF:-5} -o test.s -S  test.c -dA -fno-asynchronous-unwind-tables -fcf-protection=none
```

Let's analyze the `debug_info` section. Here we find a tree with the DWARF debug information. 

```
    .section	.debug_info,"",@progbits
.Ldebug_info0:
```

Now comes the unit header (described in section 7.5.1.1)
```
	.value	0x5	            # DWARF version number
	.byte	0x1	            # DW_UT_compile
	.byte	0x8	            # Pointer Size (in bytes)
	.long	.Ldebug_abbrev0	# Offset Into Abbrev. Section
```

## DIE entries

After that comes the DIE entries. They are identified by an uleb128 which holds the index of the referent entry in the debug_abbrev section. In the .debug_info section we 'fill' the table which is described in the .debug_abbrev section.

### `DW_TAG_compile_unit`

The first DIE entry is the compile unit.

```
	.uleb128 0x2	# (DIE (0xc) DW_TAG_compile_unit)
```

If we go to the .debug_abbrev section, we find:

```
(...)
	.uleb128 0x2	# (abbrev code)
	.uleb128 0x11	# (TAG: DW_TAG_compile_unit)
	.byte	0x1		# DW_children_yes

	.uleb128 0x25	# (DW_AT_producer)
	.uleb128 0xe	# (DW_FORM_strp)
	
	.uleb128 0x13	# (DW_AT_language)
	.uleb128 0xb	# (DW_FORM_data1)
	
	.uleb128 0x3	# (DW_AT_name)
	.uleb128 0x1f	# (DW_FORM_line_strp)
	
	.uleb128 0x1b	# (DW_AT_comp_dir)
	.uleb128 0x1f	# (DW_FORM_line_strp)
	
	.uleb128 0x11	# (DW_AT_low_pc)
	.uleb128 0x1	# (DW_FORM_addr)
	
	.uleb128 0x12	# (DW_AT_high_pc)
	.uleb128 0x7	# (DW_FORM_data8)
	
	.uleb128 0x10	# (DW_AT_stmt_list)
	.uleb128 0x17	# (DW_FORM_sec_offset)

	.byte	0
	.byte	0
(...)
```

Following the indications on Section 7.5.3, the DIE opens with the index which is used in the .debug_info section. Then follows the abbreviation code, which is one of the values of Table 7.3. 

After that we find one of `DW_children_yes` or `DW_children_no` (Table 7.4). This byte can only be one of two values:

- `DW_CHILDREN_yes` (0x01): This DIE has children. The debugger must keep reading the next DIEs as nested "descendants" of this one. The debugger needs to know when the list of children ends. DWARF does not use "length" fields for this; it uses a Null Entry (a single byte of value `0x00` in the .debug_info).

- `DW_CHILDREN_no` (0x00): This DIE is a "leaf." The very next DIE in the stream is a "sibling" (on the same level) or belongs to a parent.

Finally, we have a series of attribute specifications. These are composed of two values:

- An uleb128 which represents the attribute's name. Values listed in Table 7.5. Every attribute has its own class, which determines the form of the attribute.
- An uleb128 representing the attribute's form. The forms are described in Section 7.5.5. For example, for `string` you may choose an immediate string `DW_FORM_string` or one stored in the .debug_str section with `DW_FORM_strp`. The real values of these forms are listed in Table 7.6.

The list is closed by a null entry, with 0 in both the attribute's name and form.

We can correlate the .debug_abbrev with the .debug_info section. Notice that we fill the values from the abbrev section as if it was a table.

```
					#       v This is the offset from the start of the .debug_info section.
	.uleb128 0x2	# (DIE (0xc) DW_TAG_compile_unit)
	.long	.LASF2	# DW_AT_producer. LASF2 points to a string in the debug_str section.
	.byte	0x1d	# DW_AT_language = DW_LANG_C11 (C-11). Values listed in Table 7.17.
	.long	.LASF0	# DW_AT_name. LASF0 points to a string in the debug_str section.
	.long	.LASF1	# DW_AT_comp_dir. LASF1 points to a string in the debug_str section.
	.quad	.Ltext0	# DW_AT_low_pc. This points to the start of main().
	.quad	.Letext0-.Ltext0	# DW_AT_high_pc. End of main() minus start of main().
	.long	.Ldebug_line0	# DW_AT_stmt_list. Points to the debug_line section.
```

### `DW_TAG_subprogram`

Now for each C function we'll find a `DW_TAG_subprogram` DIE. This contains the variable declarations inside the function, the return value of the function...

Similar to before, we find an entry in the .debug_abbrev section:

```
(...)
	.uleb128 0x3	# (abbrev code)
	.uleb128 0x2e	# (TAG: DW_TAG_subprogram)
	.byte	0x1		# DW_children_yes

	.uleb128 0x3f	# (DW_AT_external)
	.uleb128 0x19	# (DW_FORM_flag_present)

	.uleb128 0x3	# (DW_AT_name)
	.uleb128 0xe	# (DW_FORM_strp)

	.uleb128 0x3a	# (DW_AT_decl_file)
	.uleb128 0xb	# (DW_FORM_data1)

	.uleb128 0x3b	# (DW_AT_decl_line)
	.uleb128 0xb	# (DW_FORM_data1)

	.uleb128 0x39	# (DW_AT_decl_column)
	.uleb128 0xb	# (DW_FORM_data1)

	.uleb128 0x49	# (DW_AT_type)
	.uleb128 0x13	# (DW_FORM_ref4)

	.uleb128 0x11	# (DW_AT_low_pc)
	.uleb128 0x1	# (DW_FORM_addr)

	.uleb128 0x12	# (DW_AT_high_pc)
	.uleb128 0x7	# (DW_FORM_data8)

	.uleb128 0x40	# (DW_AT_frame_base)
	.uleb128 0x18	# (DW_FORM_exprloc)

	.uleb128 0x7a	# (DW_AT_call_all_calls)
	.uleb128 0x19	# (DW_FORM_flag_present)

	.uleb128 0x1	# (DW_AT_sibling)
	.uleb128 0x13	# (DW_FORM_ref4)

	.byte	0
	.byte	0
(...)
```

In the .debug_info:

```
	.uleb128 0x3	# (DIE (0x2e) DW_TAG_subprogram)
	.long	.LASF3	# DW_AT_name: "main"
	.byte	0x1		# DW_AT_decl_file (test.c)
	.byte	0x5		# DW_AT_decl_line. Line where the function is declared.
	.byte	0x5		# DW_AT_decl_column. Column where the function is declared.
	.long	0x87	# DW_AT_type. 
	.quad	.LFB1	# DW_AT_low_pc. Tag to the start of main().
	.quad	.LFE1-.LFB1	# DW_AT_high_pc. Tag to the end of main() minus its start.
	.uleb128 0x1	# DW_AT_frame_base
	.byte	0x9c	# DW_OP_call_frame_cfa
			# DW_AT_call_all_calls
	.long	0x87	# DW_AT_sibling
```

### `DW_TAG_base_type`

`DW_FORM_ref4` is a 4-byte offset from the first byte of the compilation header for the compilation unit containing the reference. In the .debug_info section we find that the returning type of the function is `.long 0x87`. If we go 135 bytes (0x87 in hex) from the beginning of the .debug_info unit, we find the entry:

```
	.uleb128 0x5	# (DIE (0x87) DW_TAG_base_type)
	.byte	0x4		# DW_AT_byte_size
	.byte	0x5		# DW_AT_encoding
	.ascii "int\0"	# DW_AT_name
```

The .debug_abbrev section at 0x5:

```
	.uleb128 0x5	# (abbrev code)
	.uleb128 0x24	# (TAG: DW_TAG_base_type)
	.byte	0		# DW_children_no

	.uleb128 0xb	# (DW_AT_byte_size)
	.uleb128 0xb	# (DW_FORM_data1)
	
	.uleb128 0x3e	# (DW_AT_encoding)
	.uleb128 0xb	# (DW_FORM_data1)
	
	.uleb128 0x3	# (DW_AT_name)
	.uleb128 0x8	# (DW_FORM_string)
	
	.byte	0
	.byte	0
```

## `DW_TAG_variable`

.debug_abbrev:

```
	.uleb128 0x1	# (abbrev code)
	.uleb128 0x34	# (TAG: DW_TAG_variable)
	.byte	0	# DW_children_no

	.uleb128 0x3	# (DW_AT_name)
	.uleb128 0x8	# (DW_FORM_string)

	.uleb128 0x3a	# (DW_AT_decl_file)
	.uleb128 0x21	# (DW_FORM_implicit_const)
	.sleb128 1	# (test.c)

	.uleb128 0x3b	# (DW_AT_decl_line)
	.uleb128 0xb	# (DW_FORM_data1)

	.uleb128 0x39	# (DW_AT_decl_column)
	.uleb128 0x21	# (DW_FORM_implicit_const)
	.sleb128 9

	.uleb128 0x49	# (DW_AT_type)
	.uleb128 0x13	# (DW_FORM_ref4)

	.uleb128 0x2	# (DW_AT_location)
	.uleb128 0x18	# (DW_FORM_exprloc)

	.byte	0
	.byte	0
```

.debug_info:

```
	.uleb128 0x1	# (DIE (0x50) DW_TAG_variable)
	.ascii "x\0"	# DW_AT_name
			# DW_AT_decl_file (1, test.c)
	.byte	0x2	# DW_AT_decl_line
			# DW_AT_decl_column (0x9)
	.long	0x67	# DW_AT_type
	
	.uleb128 0x2	# DW_AT_location
	.byte	0x91	# DW_OP_fbreg
	.sleb128 -24
```
