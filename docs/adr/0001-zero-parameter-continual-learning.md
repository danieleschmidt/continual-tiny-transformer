# ADR-0001: Zero-Parameter Continual Learning Architecture

## Status
Accepted

## Context
Traditional continual learning approaches suffer from catastrophic forgetting and parameter explosion when learning multiple tasks sequentially. Each new task typically requires additional parameters, leading to linear or exponential growth in model size. Amazon Research demonstrated that transformer models can learn new tasks without adding parameters through activation-based adaptation.

## Decision
Implement a zero-parameter continual learning architecture that:
1. Freezes the base transformer weights after initial training
2. Uses task-specific activation patterns for adaptation
3. Employs elastic weight consolidation to prevent forgetting
4. Maintains constant memory footprint regardless of task count

## Consequences

### Positive
- Constant memory usage regardless of number of tasks
- No catastrophic forgetting of previous tasks
- Scalable to dozens of tasks without performance degradation
- Faster task switching due to weight reuse
- Reduced storage and deployment costs

### Negative
- Initial implementation complexity higher than naive approaches
- Requires careful tuning of activation adaptation mechanisms
- May have slight performance reduction compared to full fine-tuning
- Limited to tasks that can be handled by activation modifications

### Neutral
- Training pipeline requires task-aware scheduling
- Evaluation protocols need multi-task validation
- Documentation complexity increases due to novel approach

## Compliance
- Monitor task performance metrics across all learned tasks
- Validate memory usage remains constant during task addition
- Enforce activation-only modification policies in code reviews
- Regular benchmarking against baseline continual learning methods

## Notes
- Based on Amazon Research paper on parameter-efficient continual learning
- Inspiration from elastic weight consolidation (EWC) literature
- Related to task-specific adapter literature but with zero-parameter constraint