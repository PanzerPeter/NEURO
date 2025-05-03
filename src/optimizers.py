import torch
from torch.optim import Optimizer

class Adam(Optimizer):
    """
    Implements the Adam algorithm.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): Learning rate (default: 1e-3).
        betas (Tuple[float, float], optional): Coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999)).
        eps (float, optional): Term added to the denominator to improve numerical stability (default: 1e-8).
        weight_decay (float, optional): Weight decay (L2 penalty) (default: 0).
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            # Actual Adam update logic
            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                step = state_steps[i]
                
                # Apply weight decay (AdamW style) directly to the parameter BEFORE the main update
                # Perform stepweight decay
                if group['weight_decay'] != 0:
                    param.mul_(1 - group['lr'] * group['weight_decay'])

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1) # Use original grad here
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2) # Use original grad here
                
                # Simplified denominator calculation for clarity
                # denom = exp_avg_sq.sqrt() / (bias_correction2 ** 0.5) + group['eps']
                # More stable implementation:
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                # Calculate step size
                step_size = group['lr'] / bias_correction1
                
                # Calculate update term 
                update = exp_avg / denom

                # Apply the main Adam parameter update
                param.add_(update, alpha=-step_size)
                
                # Original update line (replaced by the two steps above for clarity and AdamW)
                # param.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

# Example Usage (optional)
if __name__ == '__main__':
    # Dummy model and data
    model = torch.nn.Linear(10, 1)
    data = torch.randn(5, 10)
    target = torch.randn(5, 1)

    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=0.01)

    print("Initial parameters:", list(model.parameters())[0].data.numpy())

    # Dummy training step
    optimizer.zero_grad()
    output = model(data)
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()

    print("Parameters after one step:", list(model.parameters())[0].data.numpy())
    print(f"Loss: {loss.item()}") 