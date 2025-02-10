import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np

from typing import Optional, Tuple, List

from CMoE_model import *

import json

import lap


@torch.no_grad()
def analyze_neuron_activations(scores: torch.Tensor,
                             K: int = 10,
                             plot_results: bool = False,
                             save_path: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:

    scores = scores.detach().cpu()
    batch_size, seq_len, inter_size = scores.shape
    total_samples = batch_size * seq_len
    
    flat_states = scores.reshape(-1, inter_size)
    activation_markers = torch.zeros_like(flat_states)
    
    for i in range(total_samples):
        sample_values = flat_states[i]
        abs_values = sample_values.abs().float()

        # Get indices of top-k absolute values
        _, top_indices = torch.topk(abs_values, k=K)
        activation_markers[i, top_indices] = 1.0
    
    # Sum up activations across all samples
    activation_counts = activation_markers.sum(dim=0)
    activation_rates = activation_counts / total_samples
    
    if plot_results:
        plt.figure(figsize=(10, 10))
        
        # Plot 1: Activation rates histogram
        plt.subplot(2, 1, 2)
        plt.hist(activation_rates.detach().to(dtype=torch.float32).numpy(), bins=500, edgecolor='black')
        plt.title('Distribution of Neuron Activation Rates')
        plt.xlabel('Activation Rate')
        plt.ylabel('Number of Neurons')
        plt.grid(True, alpha=0.3)
        
        # Add statistics for rates
        mean_rate = activation_rates.mean()
        std_rate = activation_rates.std()
        stats_text = (f'Mean rate: {mean_rate:.3f}\n'
                     f'Std rate: {std_rate:.3f}\n'
                     f'Max rate: {activation_rates.max():.3f}\n'
                     f'Min rate: {activation_rates.min():.3f}')
        
        plt.text(0.95, 0.95, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Plot 2: Neuron indices vs Activation counts
        plt.subplot(2, 1, 3)
        neuron_indices = np.arange(inter_size)
        plt.plot(neuron_indices, activation_counts.detach().to(dtype=torch.float32).numpy(), 'b-', alpha=0.6)
        plt.title('Activation Counts per Neuron Index')
        plt.xlabel('Neuron Index')
        plt.ylabel('Activation Count')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    return activation_counts, activation_rates, activation_markers


@torch.no_grad()
def construct_experts_k_means(
    activation_rates,
    activation_markers,
    num_experts,
    num_shared_experts,
    gate_dim,
    gate_proj,
    up_proj,
    down_proj):

    hidden_size = activation_rates.shape[0]
    neurons_per_expert = hidden_size // num_experts

    expert_groups = []
    remaining_indices = set(range(hidden_size))
    markers = activation_markers.float()

    _, top_indices = torch.topk(activation_rates, neurons_per_expert * num_shared_experts)
    shared_expert_indices = top_indices.tolist()
    expert_groups.append(shared_expert_indices)
    remaining_indices -= set(shared_expert_indices)
    remaining_indices_list = list(remaining_indices)

    #start_k_means

    n_samples = len(remaining_indices_list)
    k = num_experts - num_shared_experts
    cluster_size = neurons_per_expert

    remaining_rates = activation_rates[remaining_indices_list]
    _, top_indices = torch.topk(remaining_rates, k)
    selected_index = [remaining_indices_list[idx] for idx in top_indices]


    centroids = markers[:, selected_index].clone()

    max_iters = 1

    prev_assignments = None

    for iteration in range(max_iters):
        
        distances = torch.cdist(markers[:, remaining_indices_list].T, centroids.T, p=1) 
        distances_np = distances.numpy()
        repeated_distances = np.zeros((n_samples, n_samples))
        for i in range(k):
            repeated_distances[:, i*cluster_size:(i+1)*cluster_size] = distances_np[:, i:i+1]
            
        row_ind, col_ind = lap.lapjv(repeated_distances)[0:2]
        assignments = torch.tensor(col_ind // cluster_size)

        if prev_assignments is not None and torch.all(assignments == prev_assignments):
            break
        
        prev_assignments = assignments.clone()
        
        for i in range(k):
            cluster_points = markers[:, remaining_indices_list][:, assignments == i]
            if cluster_points.size(1) > 0:
                centroids[:, i] = cluster_points.mean(dim=1)
    
    representative_indices = []
    
    for i in range(k):
        cluster_mask = assignments == i
        cluster_points = markers[:, remaining_indices_list][:, cluster_mask] 
        cluster_indices = torch.where(cluster_mask)[0]
        
        distances = torch.cdist(centroids[:, i:i+1].T, cluster_points.T, p=1)
        
        closest_idx_in_cluster = torch.argmin(distances)
        original_idx = cluster_indices[closest_idx_in_cluster]
        ori_ori_idx = remaining_indices_list[original_idx]
        
        representative_indices.append(ori_ori_idx)
        group_indices = [remaining_indices_list[idx] for idx in cluster_indices]
        expert_groups.append(group_indices)
    
    experts = nn.ModuleList()
    if gate_dim is not None and up_proj is not None and down_proj is not None:
        for expert_indices in expert_groups:
            expert_mlp = LlamaMLP(gate_dim, len(expert_indices)).to('cpu')
            
            # Initialize weights from the corresponding neurons in original projections
            with torch.no_grad():
                expert_mlp.gate_proj.weight.data = gate_proj.weight.data[expert_indices, :]

                expert_mlp.up_proj.weight.data = up_proj.weight.data[expert_indices, :]
                
                expert_mlp.down_proj.weight.data = down_proj.weight.data[:, expert_indices]
            
            experts.append(expert_mlp)

    return expert_groups, experts, representative_indices



@torch.no_grad()
def construct_moe(layer, inp, attention_mask, position_ids, n_experts, n_activated, n_shared, args):

    # Forward attention
    residual = inp.to('cuda')
    hidden_states = layer.input_layernorm(inp.to('cuda'))
    hidden_states = layer.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids)[0]
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)

    # Forward FFN, normalize FFN input and weights to compute normalized h
    h_pre = hidden_states
    h = layer.mlp.act_fn(F.linear(F.normalize(hidden_states, p=2, dim=-1), F.normalize(layer.mlp.gate_proj.weight, p=2, dim=1)))
    h = h * F.linear(F.normalize(hidden_states, p=2, dim=-1), F.normalize(layer.mlp.up_proj.weight, p=2, dim=1))

    scores_all = h.to('cpu')

    # Calculate activation markers, activation rates
    counts, rates, markers = analyze_neuron_activations(scores_all, K=args.k_act, plot_results=False, save_path=f'./plot/scores_analysis_{layer.self_attn.layer_idx}.png')

    # Construct shared and routed experts
    expert_groups, experts, representative_indices = construct_experts_k_means(
        rates,
        markers,
        num_experts=n_experts,
        num_shared_experts = n_shared,
        gate_dim=layer.mlp.hidden_size,  # Same as original MLP hidden dim
        gate_proj=layer.mlp.gate_proj,  # Original gate projection
        up_proj=layer.mlp.up_proj,  # Original up projection
        down_proj=layer.mlp.down_proj,  # Original down projection
    )

    
    # Router
    router = Router(layer.mlp.hidden_size, n_experts-n_shared, n_activated, bias_speed = args.bias_speed)

    core_neurons = representative_indices
    core_weights = layer.mlp.up_proj.weight.data[core_neurons, :]
    core_gate_weights = layer.mlp.gate_proj.weight.data[core_neurons, :]
    router.classifier.weight.data = F.normalize(core_weights, p=2, dim=1)
    router.gate.weight.data = F.normalize(core_gate_weights, p=2, dim=1)

    # MoE
    moe = MoE(layer.mlp.hidden_size, layer.mlp.intermediate_size//n_experts, n_experts, n_shared, n_activated)
    moe.gate = router
    moe.experts = experts[1:]
    moe.shared_experts = experts[0]
    moe.cus_training = False
    moe_out = moe(h_pre) + residual


    layer.mlp = moe

    return moe_out
