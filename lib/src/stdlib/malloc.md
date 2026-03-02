# A brief explanation on the malloc algorithm

This `malloc` implementation only works for single threaded applications. It has been optimized to be relatively fast and use as little space as posible. Further refinement can be done by tweaking the definitions in [stdlib.h](../../headers/stdlib.h).

## How is data represented?

Memory is splitted in **heap chunks**. If there's no free space on any of the previously requested chunks, this implementation requests the OS for new memory in multiples of `HEAP_CHUNK_SIZE`. Its size depends on the amount of memory requested.

Each heap chunk has its `HeapChunkHeader`, which is used to store:
- The size of the chunk.
- The used space of the chunk. Space is used by blocks which aren'f freed.
- A pointer to the last free block in the chunk.

Inside every heap chunk there are **memory blocks**, each one starts with a `BlockHeader`.  Inside the header we have:
- Pointers to the next and previously freed block. When the block is freed, it is inserted into a linked list which stores the blocks that have been freed. The order of the list matters. Blocks from the same heap are stored together in this list. Blocks of the same heap are sorted from least to greatest.
- The size of the block, aligned to `BLOCK_ALIGNMENT`.
- A pointer to the heap chunk to which the block belongs to.
- An assortment of flags. `isInUse` marks the block as used by the user. An used block cannot be deallocated.

We also have a list of `HeapChunkHeader*` which stores empty chunks (`chunkDeallocateList`). When this list is full, the oldest chunk in the list (at index 0) is deallocated by the OS.

## The algorithm

When `malloc` is called:
- Is there any free block with a size greater or equal to the requested block? 
   - If so, check the size of the free block: if you add your block, is there enough space for a second free block? 
      - If so, split the free block. The first part will be used for your current block, the second will be left free.
      - If there's not enough space for the second block, modify the size of the requested block to fill the whole free block.
    - Create the block by occupying the free block.
- If there wasn't any free block, request the OS for a new heap chunk.  
  - Is there enough space for another block in this heap?
    - If so, create a free block positioned after the requested block.
    - If not, modify the size of the block to fill the whole chunk.
  - Create your block in this heap chunk.

When `free` is called:
- Add the block to the free list.
- Is the block to be freed surrounded by any other free blocks? 
  - If so, try to coalesce them. They must be in the same heap chunk and be adjacent in memory too.
- Is the current chunk empty?
  - Is `chunkDeallocateList` completely filled?
    - If so, deallocate the oldest block in the list (the one at index 0). 
  - Add the heap chunk to the `chunkDeallocateList`. 