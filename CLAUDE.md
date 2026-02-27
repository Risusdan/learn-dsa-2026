# CLAUDE.md — learn-dsa-2026

## Project Structure

This is a self-study curriculum for learning Data Structures & Algorithms.

- `README.md` — Repository overview
- `LearningPath.md` — Full topic tree with OPTIONAL markers
- `Schedule.md` — Week-by-week plan with checkboxes and "Why It Matters" context
- `weeks/week-NN/topic-name.md` — Individual lesson files (generated over time)

## How to Generate Lessons

1. **Always read `Schedule.md` first** to determine what topics to cover for the requested week.
2. Only cover topics listed in the schedule. Respect OPTIONAL markers in `LearningPath.md` — do not include OPTIONAL topics unless explicitly asked.
3. Create lesson files at `weeks/week-NN/topic-name.md` (e.g., `weeks/week-04/sorting-algorithms.md`).
4. One file per major topic grouping within a week. If a week covers multiple distinct topics, split them into separate files.
5. **Update `README.md`** after generating lessons — add or update a `## Lessons` section with links grouped by week. Use the week title from `Schedule.md`:
   ```markdown
   ## Lessons

   ### Week 1 — [Week Title from Schedule]
   - [Lesson Title](weeks/week-01/lesson-name.md)
   ```

## Lesson File Template

Every lesson file must follow this exact structure:

```markdown
# Topic Name

> One-sentence summary of what this topic is and why it matters.

## Table of Contents
- [Core Concepts](#core-concepts)
- [Code Examples](#code-examples)
- [Common Pitfalls](#common-pitfalls)
- [Key Takeaways](#key-takeaways)

## Core Concepts

[Explanation broken into logical subsections with ### headings as needed.
For every concept, cover three layers:
1. WHAT — what is this thing?
2. HOW — how does it work / how do you use it?
3. WHY — why does it exist? Why this approach over alternatives? Why does it matter?
The "why" is the most important layer — it builds lasting intuition.
Use `####` subheadings (What / How / Why It Matters) under each `###` concept heading
to keep the three layers visually distinct and scannable.
Do NOT use inline bold labels like "**What.**" — they blend into paragraph text.
Build intuition first, then add precision. Use analogies for abstract ideas.
Keep paragraphs short (3-5 sentences max).
Use Mermaid diagrams (```mermaid blocks) when visual representation helps —
e.g., data structure layouts, tree/graph structures, algorithm step-by-step flow,
pointer movements, partition steps.]

## Code Examples

[Annotated, idiomatic, production-quality code. Show how a professional
would actually write this — proper naming, error handling where appropriate,
clean structure. Each example should be self-contained and runnable.]

## Common Pitfalls

[Bad vs good code comparisons. Each pitfall gets:
- What the mistake is
- Why it's wrong
- The correct approach with code]

## Key Takeaways

- Bullet list of 3-5 most important points
- What to remember, what to internalize
```

## Writing Style

- **Audience**: Self-learner studying independently — no instructor, no classroom.
- **Tone**: Concise, expert, opinionated. Write like a senior engineer mentoring a colleague, not a textbook.
- **Structure**: Build intuition first, then add precision. Use analogies for abstract ideas.
- **Why-first**: For every concept, always explain *why* it exists, *why* this approach, and *why* it matters. The "why" is more important than the "what."
- **Paragraphs**: Keep short — 3-5 sentences max. Dense walls of text kill learning.
- **No exercises**: Do not include practice problems, homework, or challenges.
- **No external resources**: Do not include "Further Reading" sections or links to external material.

## Code Example Standards

- Write **idiomatic, production-quality code** — the kind a senior engineer would write at work.
- Show professional coding habits: meaningful names, type hints, proper error handling, clean structure.
- Every code block must be **self-contained and runnable** (include necessary imports).
- Use detailed inline comments to explain *why*, not just *what*.
- When showing a pattern or technique, show the **realistic use case**, not a toy example.

## Common Pitfalls Format

Each pitfall must include:

1. A brief description of the mistake
2. A `# BAD` code block showing the wrong way
3. An explanation of *why* it's wrong
4. A `# GOOD` code block showing the correct approach

```python
# BAD — [description of what's wrong]
[bad code]

# GOOD — [description of the fix]
[good code]
```

---

## Repo-Specific: Data Structures & Algorithms (Python)

### Language & Version

- **Language**: Python (all code examples in Python)
- **Target version**: Python 3.12+ (use modern syntax and features)

### Coding Conventions (PEP 8 and beyond)

- Follow **PEP 8** strictly — `snake_case` for variables/functions, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants
- **Type hints everywhere** — every function signature must have full type annotations (parameters and return type)
- Use `str | None` syntax (PEP 604) over `Optional[str]`
- Use **f-strings** for all string formatting — never `%` or `.format()`
- Use `dataclasses` for node/graph/tree structures where appropriate

### DSA-Specific Requirements

#### Big-O Analysis — Mandatory for Every Solution

Every algorithm and data structure operation **must** include complexity analysis:

- **Time complexity**: Always state Big-O for worst case. Include average case when it differs meaningfully (e.g., quicksort O(n log n) average vs O(n²) worst).
- **Space complexity**: Always state Big-O for auxiliary space (excluding input).
- Place complexity in a clear, consistent format after each implementation:

```
**Time**: O(n log n) — we divide the array in half each level (log n levels) and merge n elements per level.
**Space**: O(n) — merge step requires a temporary array of size n.
```

- Don't just state the complexity — **explain *why* it's that complexity** in one sentence.

#### Brute-Force → Optimized Progression

For problems where optimization is the point (searching, DP, sliding window, etc.):

1. **Start with the brute-force solution** — show the naive approach first and analyze its complexity.
2. **Identify the bottleneck** — explain *what* makes the brute-force slow and *why*.
3. **Show the optimized solution** — introduce the better algorithm/data structure and explain the insight that enables the improvement.
4. **Compare complexities** side by side.

This progression teaches *how to think about optimization*, not just the final answer.

#### Mermaid Diagrams for Data Structures

Use Mermaid diagrams liberally for:

- **Data structure layouts** — linked lists (nodes with pointers), trees (node relationships), graphs (adjacency)
- **Algorithm walkthroughs** — step-by-step state changes (e.g., how a partition step works in quicksort, how BFS explores level by level)
- **Pointer movements** — two-pointer technique, slow/fast pointers, sliding window boundaries

Mermaid is especially valuable here because DSA concepts are inherently visual. A diagram of a tree rotation or a BFS traversal order communicates more than paragraphs of text.

#### Python DSA Conventions

- Use Python's built-in structures where appropriate: `list` as dynamic array, `dict` as hash map, `collections.deque` as queue, `heapq` for heaps
- When implementing a data structure from scratch (e.g., linked list, BST), use `dataclass` for node types
- For graph problems, show adjacency list representation using `dict[str, list[str]]` or `defaultdict(list)`
- Use `float('inf')` for infinity sentinel values in graph algorithms
- Show both recursive and iterative approaches for tree/graph traversals when both are practical
