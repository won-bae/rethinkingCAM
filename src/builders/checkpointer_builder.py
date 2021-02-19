from src.core.checkpointers import CustomCheckpointer

def build(mode, train_dir, model, log, optimizer, scheduler, eval_standard):
    checkpointer = CustomCheckpointer(
        mode, train_dir, model, log, optimizer, scheduler, eval_standard)
    return checkpointer

