import shutil
from torch.utils.tensorboard import SummaryWriter

def get_summary_writer(run_dir: str = "logs", delete_logs=True) -> SummaryWriter:
    """Get a base TensorBoard summary writer.

    Args:
        run_dir (str): The directory where the TensorBoard logs will be saved.

    Returns:
        SummaryWriter: A TensorBoard summary writer.
    """
    if delete_logs:
        shutil.rmtree(run_dir, ignore_errors=True)
    
    return SummaryWriter(log_dir=run_dir)