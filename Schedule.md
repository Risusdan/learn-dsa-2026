# Data Structures & Algorithms Learning Schedule

**Scope**: All required topics from the [Learning Path](LearningPath.md) (OPTIONAL topics excluded)

## Progress Tracker

| Week | Topic                                          | Status         |
| ---- | ---------------------------------------------- | -------------- |
| 1    | Intro + Arrays + Linked Lists                  | ✅ Done        |
| 2    | Stacks, Queues, Hash Tables                    | ✅ Done        |
| 3    | Algorithmic Complexity                         | ✅ Done        |
| 4    | Sorting Algorithms                             | ✅ Done        |
| 5    | Searching + Two Pointers + Sliding Window      | ✅ Done        |
| 6    | Recursion, Backtracking, Divide and Conquer    | ✅ Done        |
| 7    | Trees                                          | ✅ Done        |
| 8    | Balanced Trees + Heaps                         | ✅ Done        |
| 9    | Graphs                                         | ✅ Done        |
| 10   | Graph Algorithms                               | ✅ Done        |
| 11   | Dynamic Programming + Greedy Algorithms        | ✅ Done        |
| 12   | Advanced DS + Remaining Techniques + Practice  | ✅ Done        |

---

## Week 1: Intro + Arrays + Linked Lists

*Review week — skim quickly if you already know this material.*

- [ ] What are Data Structures? — ways to organize, store, and access data efficiently
- [ ] Why are Data Structures Important? — performance, scalability, problem-solving foundation
- [ ] Arrays — contiguous memory, O(1) access by index, O(n) insertion/deletion
- [ ] Linked Lists — nodes with pointers, O(1) insertion/deletion at head, O(n) access by index
- [ ] Singly vs Doubly Linked Lists — trade-offs between memory and traversal flexibility

**Why It Matters**: Arrays and linked lists are the two fundamental ways to store sequences — one uses contiguous memory (fast access, slow insertion) and the other uses pointers (slow access, fast insertion). Every other data structure is built on top of these two primitives. Understanding *why* arrays have O(1) access (pointer arithmetic) and linked lists have O(1) insertion (pointer rewiring) teaches you to reason about performance from first principles.

---

## Week 2: Stacks, Queues, Hash Tables

- [ ] Stacks — LIFO (Last In, First Out), `push`, `pop`, `peek`
- [ ] Stack applications — function call stack, expression evaluation, undo operations
- [ ] Queues — FIFO (First In, First Out), `enqueue`, `dequeue`
- [ ] Queue applications — BFS, task scheduling, buffering
- [ ] Hash Tables — key-value storage, hash functions, collision resolution (chaining, open addressing)
- [ ] Hash table performance — O(1) average case, O(n) worst case, load factor and resizing

**Why It Matters**: Hash tables are arguably the most important data structure in practical programming — Python's `dict`, JavaScript's objects, and Java's `HashMap` are all hash tables. Understanding *how* they achieve O(1) average-case lookups (hashing keys to array indices) and *when* they degrade (hash collisions, poor hash functions) is essential. Stacks and queues appear everywhere: the call stack is literally a stack, and BFS uses a queue.

---

## Week 3: Algorithmic Complexity

- [ ] Time vs Space Complexity — measuring algorithm efficiency in terms of input size
- [ ] How to Calculate Complexity — counting operations, identifying dominant terms
- [ ] Big-O Notation — upper bound (worst case), the most commonly used notation
- [ ] Big-θ Notation — tight bound (exact growth rate)
- [ ] Big-Ω Notation — lower bound (best case)
- [ ] Common Runtimes — O(1) constant, O(log n) logarithmic, O(n) linear, O(n log n), O(n²) polynomial, O(2ⁿ) exponential, O(n!) factorial

**Why It Matters**: Big-O is the language engineers use to discuss performance. When someone says "binary search is O(log n)," they're saying the number of steps grows logarithmically with input size — doubling the data adds only one more step. Without this framework, you can't meaningfully compare algorithms, predict how code will behave at scale, or explain performance decisions in interviews. It's the foundation of every algorithmic discussion that follows.

---

## Week 4: Sorting Algorithms

- [ ] Bubble Sort — compare adjacent pairs, swap if out of order, O(n²)
- [ ] Insertion Sort — build sorted array one element at a time, O(n²), good for small/nearly-sorted data
- [ ] Selection Sort — find minimum, place at front, O(n²)
- [ ] Merge Sort — divide and conquer, O(n log n), stable, requires O(n) extra space
- [ ] Quick Sort — pivot partitioning, O(n log n) average, O(n²) worst case, in-place
- [ ] Heap Sort — build a heap, extract max repeatedly, O(n log n), in-place

**Why It Matters**: Sorting is the most-studied problem in computer science because it appears everywhere — databases, search results, scheduling, rendering. Knowing *why* O(n log n) is the theoretical lower bound for comparison sorts, and understanding the trade-offs (Merge Sort is stable but uses extra memory, Quick Sort is in-place but has O(n²) worst case) lets you choose the right algorithm for your constraints. In practice, most languages use hybrid sorts (Timsort, Introsort) that combine the best properties.

---

## Week 5: Searching + Two Pointers + Sliding Window

- [ ] Linear Search — check every element, O(n), works on unsorted data
- [ ] Binary Search — divide search space in half each step, O(log n), requires sorted data
- [ ] Binary search variations — finding first/last occurrence, search in rotated array
- [ ] Two Pointer Technique — one pointer at each end, moving inward based on conditions
- [ ] Two pointer applications — pair sum, palindrome checking, removing duplicates
- [ ] Sliding Window Technique — fixed or variable-size window sliding across data
- [ ] Sliding window applications — maximum subarray sum, longest substring without repeating characters

**Why It Matters**: Binary search is the most powerful technique for reducing O(n) to O(log n) — but it applies far beyond sorted arrays (binary search on answer, search in monotonic functions). Two pointers and sliding window are the two most common patterns for solving array/string problems efficiently. They transform brute-force O(n²) solutions into O(n) by maintaining a smart traversal state. These three techniques alone solve a huge proportion of interview problems.

---

## Week 6: Recursion, Backtracking, Divide and Conquer

- [ ] Recursion — base case + recursive case, call stack mechanics, stack overflow risks
- [ ] Recursive thinking — breaking problems into identical sub-problems
- [ ] Backtracking — try all options, undo (backtrack) when a path fails
- [ ] Backtracking applications — N-Queens, Sudoku solver, permutations, combinations
- [ ] Divide and Conquer — split problem in half, solve each half, combine results
- [ ] Divide and conquer applications — merge sort, binary search, closest pair of points

**Why It Matters**: Recursion is the natural way to express problems with self-similar structure (trees, nested data, mathematical sequences). Backtracking extends recursion to explore all possibilities systematically — it's the foundation of constraint satisfaction, puzzle solving, and combinatorial optimization. Divide and conquer is the strategy behind the most efficient algorithms (merge sort, FFT, Strassen's matrix multiplication). Mastering these three patterns gives you a toolkit for solving problems that seem intractable at first glance.

---

## Week 7: Trees

- [ ] Binary Trees — each node has at most two children, terminology (root, leaf, height, depth)
- [ ] Binary Search Trees (BST) — left < parent < right, O(log n) search/insert/delete (balanced)
- [ ] Tree Traversal — In-Order (sorted output for BST), Pre-Order, Post-Order
- [ ] Breadth First Search (BFS) — level-by-level traversal using a queue
- [ ] Depth First Search (DFS) — explore as deep as possible before backtracking, using stack/recursion

**Why It Matters**: Trees are the most versatile data structure in computer science. File systems are trees. The DOM is a tree. Database indices are trees. JSON is a tree. Understanding tree traversals is fundamental — in-order traversal of a BST gives sorted output, BFS finds the shortest path in unweighted graphs, and DFS is the basis for topological sort, cycle detection, and connected components. Trees are also the most common topic in coding interviews.

---

## Week 8: Balanced Trees + Heaps

- [ ] AVL Trees — self-balancing BST, rotations (left, right, left-right, right-left)
- [ ] Why balancing matters — unbalanced BST degrades to O(n), balanced guarantees O(log n)
- [ ] B-Trees — multi-way balanced trees, used in databases and file systems
- [ ] Why B-Trees exist — minimize disk I/O by maximizing keys per node
- [ ] Heaps — complete binary tree, min-heap and max-heap property
- [ ] Heap operations — insert (bubble up), extract-min/max (bubble down), heapify
- [ ] Heap applications — priority queues, heap sort, finding k-th largest element

**Why It Matters**: Balanced trees solve the fundamental problem of BSTs — a sorted sequence inserted into a BST creates a linked list with O(n) operations. AVL trees guarantee O(log n) by rebalancing after every insertion. B-Trees are optimized for storage systems where reading a block of data is cheap but seeking is expensive (hard drives, SSDs). Heaps give O(1) access to the min/max element and O(log n) insertion — the foundation of priority queues used in Dijkstra's algorithm, task schedulers, and event-driven systems.

---

## Week 9: Graphs

- [ ] Directed Graphs — edges have direction, represent dependencies, flow, hierarchy
- [ ] Undirected Graphs — edges are bidirectional, represent relationships, connections
- [ ] Graph representations — adjacency matrix vs adjacency list, trade-offs
- [ ] BFS on Graphs — shortest path in unweighted graphs, level-order exploration
- [ ] DFS on Graphs — cycle detection, topological sort, connected components
- [ ] Topological Sort — ordering tasks with dependencies, only for DAGs

**Why It Matters**: Graphs model relationships — social networks (who follows whom), road networks (cities and routes), dependency graphs (build systems, package managers), and state machines (game logic, UI flows). BFS finds the shortest path in unweighted graphs (GPS navigation with equal-weight roads). DFS detects cycles (circular dependencies in build systems) and performs topological sort (determining build order). Learning to see real-world problems as graph problems is a key algorithmic skill.

---

## Week 10: Graph Algorithms

- [ ] Dijkstra's Algorithm — shortest path from source to all vertices, non-negative weights, O(V + E log V) with min-heap
- [ ] Bellman-Ford Algorithm — shortest path with negative weights, detects negative cycles, O(V × E)
- [ ] A* Algorithm — heuristic-guided shortest path, optimal when heuristic is admissible
- [ ] Minimum Spanning Tree — connecting all vertices with minimum total weight
- [ ] Prim's Algorithm — grow MST from a starting vertex, greedy, O(V + E log V) with min-heap
- [ ] Kruskal's Algorithm — sort edges by weight, add if no cycle (using Union-Find), O(E log E)

**Why It Matters**: Dijkstra's algorithm powers GPS navigation and network routing (OSPF protocol). Bellman-Ford handles negative weights (financial arbitrage detection, currency exchange). A* is the standard for game pathfinding and robotics. Minimum spanning trees solve network design problems — connecting cities with minimum cable, designing circuit boards, clustering data. These algorithms appear in system design interviews and real-world infrastructure.

---

## Week 11: Dynamic Programming + Greedy Algorithms

- [ ] Dynamic Programming — breaking problems into overlapping subproblems, storing solutions
- [ ] Memoization (top-down) — recursive with cache, natural to write
- [ ] Tabulation (bottom-up) — iterative with table, often more space-efficient
- [ ] Classic DP problems — Fibonacci, knapsack, longest common subsequence, coin change
- [ ] Greedy Algorithms — making locally optimal choices at each step
- [ ] When greedy works — optimal substructure + greedy choice property
- [ ] Classic greedy problems — activity selection, Huffman coding, fractional knapsack

**Why It Matters**: Dynamic programming is considered the hardest algorithmic technique because it requires recognizing overlapping subproblems — the same subproblem being solved repeatedly. The key insight is that you only solve each subproblem once and store the result. Greedy algorithms are simpler but only work when the greedy choice property holds (local optimality leads to global optimality). Knowing *when* greedy works (and proving it) versus when you need DP is a critical skill. These two techniques together solve the majority of optimization problems.

---

## Week 12: Advanced DS + Remaining Techniques + Practice

- [ ] Trie — prefix tree, O(L) insert/search where L is word length
- [ ] Trie applications — autocomplete, spell checking, IP routing tables
- [ ] Disjoint Set (Union-Find) — track connected components, `find` with path compression, `union` by rank
- [ ] Union-Find applications — Kruskal's MST, detecting cycles, connected components
- [ ] Brute Force — exhaustive search, establishing correctness before optimizing
- [ ] Kth Element — quick-select algorithm, O(n) average for finding k-th smallest/largest
- [ ] Island Traversal — grid-based BFS/DFS for connected region problems
- [ ] Two Heaps — median maintenance using a max-heap and min-heap
- [ ] Merge Intervals — sorting by start time, merging overlapping intervals
- [ ] Cyclic Sort — placing elements at correct indices, O(n) for arrays with values 1–n
- [ ] Fast and Slow Pointers — cycle detection (Floyd's algorithm), finding middle of linked list
- [ ] Leetcode — practice platform for applying all techniques

**Why It Matters**: Tries and Union-Find are specialized but powerful — tries power every search engine's autocomplete, and Union-Find is the most efficient way to track connected components (used in Kruskal's algorithm and network connectivity). The remaining techniques (two heaps, merge intervals, cyclic sort, fast/slow pointers) are common problem-solving patterns that appear repeatedly in interview problems and real systems. This week consolidates everything into practical problem-solving ability.

---
up:: [MOC-Programming](../../../01-index/MOC-Programming.md)
#type/learning #source/self-study #status/seed
