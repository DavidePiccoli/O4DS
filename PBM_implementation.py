import cvxpy as cp
import cplex

import torch
import torch.nn as nn
import torch.optim as optim
import time

# Proximal Bundle Method with Subproblem
def proximal_bundle_method_manual(model, dataloader, max_iter=100, t=10.0, m_1=1, max_sub=50, l1_lambda=0.01, t_factor = 0.0001, adaptive=True):

    bundle = []  #(weights, loss, gradients) tuples
    history = []  # Loss history
    subproblem_time = 0.0
    error_count = 0

    for epoch in range(max_iter):
        epoch_loss = 0.0

        for inputs, targets in dataloader:
            # Forward
            outputs = model.forward(inputs)

            # MSE loss
            mse_loss = ((outputs - targets) ** 2).mean()

            # L1 regularization
            l1_loss = l1_lambda * (torch.sum(torch.abs(model.w1)) + torch.sum(torch.abs(model.w2)))
            total_loss = mse_loss + l1_loss
            epoch_loss = total_loss.item()

            # backward pass
            model.backward(inputs, targets, outputs, l1_lambda=l1_lambda)

            # Store current gradients and weights in the bundle
            gradients = [model.dL_dw1.clone(), model.dL_db1.clone(), model.dL_dw2.clone(), model.dL_db2.clone()]
            weights = [model.w1.clone(), model.b1.clone(), model.w2.clone(), model.b2.clone()]
            bundle.append((weights, total_loss.item(), gradients))

            # FIFO limited-memory
            if len(bundle) > max_sub:
                bundle.pop(0)

            # Subproblem
            z_star = []
            for param in [model.w1, model.b1, model.w2, model.b2]:
                z_star.append(cp.Variable(param.shape))

            obj_terms = []
            for w_i, f_i, g_i in bundle:
                linear_approximation = 0
                for z, g, w in zip(z_star, g_i, w_i):
                    linear_approximation += cp.sum(cp.multiply(g.detach().numpy(), z - w.detach().numpy()))
                obj_terms.append(f_i + linear_approximation)

            objective = cp.Minimize(cp.max(cp.hstack(obj_terms)) + (1 / (2 * t)) * sum(cp.sum_squares(z - torch.tensor(w_i).numpy()) for z, w_i in zip(z_star, weights)))

            start_time = time.time()
            max_errors = 5
            try:
                problem = cp.Problem(objective)
                problem.solve(solver=cp.CPLEX, verbose=False, warm_start=True)   #CPLEX solver
            except Exception as e:
                print(f"CPLEX solver failed at epoch {epoch+1}")
                #print('objective:', objective)
                error_count += 1
                if error_count >= max_errors:
                    print("stopping training")
                    end_time = time.time()
                    time_to_add = end_time - start_time
                    subproblem_time += time_to_add
                    break 
                continue  # Skip only this iteration
            end_time = time.time()

            time_to_add = end_time - start_time
            subproblem_time += time_to_add

            # Compute f_B(x + d^*) as the max approximation in the bundle
            f_B_x_d_star = max(
                f_i + sum((g @ (w_new - w).T).sum().value if (g @ (w_new - w).T).sum().value is not None else 0 for g, w_new, w in zip(g_i, z_star, w_i) if g.shape == (w_new - w).shape)
                for w_i, f_i, g_i in bundle)

            # Update model parameters
            for param, z in zip([model.w1, model.b1, model.w2, model.b2], z_star):
                value = z.value if z.value is not None else 0

                param.copy_(torch.tensor(value, dtype=torch.float32))

            # Evaluate the candidate solution
            outputs_candidate = model.forward(inputs)
            mse_loss_candidate = ((outputs_candidate - targets) ** 2).mean()
            l1_loss_candidate = l1_lambda * (torch.sum(torch.abs(model.w1)) + torch.sum(torch.abs(model.w2)))
            total_loss_candidate = mse_loss_candidate + l1_loss_candidate

            # Check descent condition
            if epoch>=5 and (total_loss_candidate.item() - total_loss.item()) <= m_1 * (f_B_x_d_star - total_loss.item()):
                if adaptive:
                    t *= 1-t_factor
                print(f'Epoch {epoch+1}, DESCENT step, Loss: {total_loss_candidate.item():.4f}', 't:', t)
            else:
                if adaptive:
                    t *= 1+t_factor
                print(f'Epoch {epoch+1}, NULL step, Loss: {total_loss_candidate.item():.4f}', 't:', t)  # Null step

        # Record loss for the epoch
        history.append(epoch_loss / len(dataloader))
        print(f'Epoch {epoch+1}/{max_iter}, Avg Loss: {history[-1]:.4f}')

    return model, history, subproblem_time


# 2 hidden layers version
def proximal_bundle_method_manual_2l(model, dataloader, max_iter=100, t=10.0, m_1=1, max_sub=50, l1_lambda=0.01, t_factor=0.0001, adaptive=True):

    bundle = []  #(weights, loss, gradients) tuples
    history = []  # Loss history
    subproblem_time = 0.0
    error_count = 0

    for epoch in range(max_iter):
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            # Forward
            outputs = model.forward(inputs)
            
            #MSE loss
            mse_loss = ((outputs - targets) ** 2).mean()
            
            # Add L1 regularization
            l1_loss = l1_lambda * (torch.sum(torch.abs(model.w1)) + torch.sum(torch.abs(model.w2)) + torch.sum(torch.abs(model.w3)))
            total_loss = mse_loss + l1_loss
            epoch_loss = total_loss.item()
            
            # backward pass
            model.backward(inputs, targets, outputs, l1_lambda=l1_lambda)
            
            # Store current gradients and weights in the bundle
            gradients = [
                model.dL_dw1.clone(), model.dL_db1.clone(),
                model.dL_dw2.clone(), model.dL_db2.clone(),
                model.dL_dw3.clone(), model.dL_db3.clone()
            ]
            weights = [
                model.w1.clone(), model.b1.clone(),
                model.w2.clone(), model.b2.clone(),
                model.w3.clone(), model.b3.clone()
            ]
            bundle.append((weights, total_loss.item(), gradients))
            
            # FIFO limited-memory
            if len(bundle) > max_sub:
                bundle.pop(0)
            
            # Subproblem
            z_star = [cp.Variable(param.shape) for param in [model.w1, model.b1, model.w2, model.b2, model.w3, model.b3]]
            
            obj_terms = []
            for w_i, f_i, g_i in bundle:
                linear_approximation = sum(cp.sum(cp.multiply(g.detach().numpy(), z - w.detach().numpy())) for z, g, w in zip(z_star, g_i, w_i))
                obj_terms.append(f_i + linear_approximation)
            
            objective = cp.Minimize(cp.max(cp.hstack(obj_terms)) + (1 / (2 * t)) * sum(cp.sum_squares(z - torch.tensor(w_i).numpy()) for z, w_i in zip(z_star, weights)))
            
            start_time = time.time()
            max_errors = 5
            try:
                problem = cp.Problem(objective)
                problem.solve(solver=cp.CPLEX, verbose=False, warm_start=True)   #CPLEX solver
            except Exception as e:
                print(f"CPLEX solver failed at epoch {epoch+1}")
                #print('objective:', objective)
                error_count += 1
                if error_count >= max_errors:
                    print("stopping training")
                    end_time = time.time()
                    time_to_add = end_time - start_time
                    subproblem_time += time_to_add
                    break 
                continue  # Skip only this iteration
            end_time = time.time()

            time_to_add = end_time - start_time
            subproblem_time += time_to_add

            # Compute f_B(x + d^*) as the max approximation in the bundle
            f_B_x_d_star = max(
                f_i + sum((g @ (w_new - w).T).sum().value if (g @ (w_new - w).T).sum().value is not None else 0 for g, w_new, w in zip(g_i, z_star, w_i) if g.shape == (w_new - w).shape)
                for w_i, f_i, g_i in bundle)
            
            # Update model parameters
            for param, z in zip([model.w1, model.b1, model.w2, model.b2, model.w3, model.b3], z_star):
                param.copy_(torch.tensor(z.value if z.value is not None else 0, dtype=torch.float32))
            
            # Evaluate the candidate solution
            outputs_candidate = model.forward(inputs)
            mse_loss_candidate = ((outputs_candidate - targets) ** 2).mean()
            l1_loss_candidate = l1_lambda * (torch.sum(torch.abs(model.w1)) + torch.sum(torch.abs(model.w2)) + torch.sum(torch.abs(model.w3)))
            total_loss_candidate = mse_loss_candidate + l1_loss_candidate
            
            # Check descent condition
            if epoch>=5 and (total_loss_candidate.item() - total_loss.item()) <= m_1 * (f_B_x_d_star - total_loss.item()):
                if adaptive:
                    t *= 1-t_factor
                print(f'Epoch {epoch+1}, DESCENT step, Loss: {total_loss_candidate.item():.4f}', 't:', t)
            else:
                if adaptive:
                    t *= 1+t_factor
                print(f'Epoch {epoch+1}, NULL step, Loss: {total_loss_candidate.item():.4f}', 't:', t)  # Null step
        
        history.append(epoch_loss / len(dataloader))
        print(f'Epoch {epoch+1}/{max_iter}, Avg Loss: {history[-1]:.4f}')
    
    return model, history, subproblem_time