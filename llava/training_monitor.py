from matplotlib import pyplot as plt


class TrainingMonitor:
    def __init__(self, log_interval=50):
        self.log_interval = log_interval
        self.step = 0

        # 存储所有指标
        self.metrics = {
            'loss': [],
            'action_accuracy': [],
            'l1_loss': [],
            'steps': []
        }

    def record_step(self, loss, action_accuracy, l1_loss, gradient_step_idx=None):
        """记录训练步骤"""
        self.step += 1

        if gradient_step_idx is None:
            gradient_step_idx = self.step

        self.metrics['loss'].append(float(loss))
        self.metrics['action_accuracy'].append(float(action_accuracy))
        self.metrics['l1_loss'].append(float(l1_loss))
        self.metrics['steps'].append(gradient_step_idx)

        # 定期打印状态
        if gradient_step_idx % self.log_interval == 0:
            self._print_status(gradient_step_idx, loss, action_accuracy, l1_loss)

    def _print_status(self, step, loss, accuracy, l1_loss):
        """打印当前状态"""
        print(f"步骤 {step:6d} | "
              f"Loss: {loss:8.8f} | "
              f"准确率: {accuracy:6.2%} | "
              f"L1误差: {l1_loss:8.4f}")

    def plot_metrics(self, save_path=None):
        """绘制所有指标"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

        # Loss曲线
        ax1.plot(self.metrics['steps'], self.metrics['loss'])
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Gradient Step')
        ax1.set_ylabel('Loss')
        ax1.grid(True)

        # 准确率曲线
        ax2.plot(self.metrics['steps'], self.metrics['action_accuracy'])
        ax2.set_title('Action Accuracy')
        ax2.set_xlabel('Gradient Step')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)

        # L1损失曲线
        ax3.plot(self.metrics['steps'], self.metrics['l1_loss'])
        ax3.set_title('Action L1 Loss')
        ax3.set_xlabel('Gradient Step')
        ax3.set_ylabel('L1 Loss')
        ax3.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()