# Policy based methods

Goal is to maximize the true value function of a parameterized policy from all initial states.
So maximize the true value function by changing the policy (without touching the value function).  
  
We want to **find the gradient** which will help reaching this objective

## REINFORCE
<center>Use of function approximation (here a policy network) to generate probabilities over actions</center>  

<style>
    svg[id^="mermaid-"] { width: 100%; max-height: 450px;}
</style>
  ```mermaid

  %%{ init: { 'flowchart': { 'curve': 'basis' } } }%%
graph TD;
    subgraph Policy Network
    id1((s1)) & id2((s2)) & id3((s3)) & id4((s4)) ---> 
    h1((h1)) & h2((h2)) & h3((h3)) & h4((h4)) & h5((h5)) & hn((hn)) ---> a1((a1)) & a2((a2));
end
```
The Policy Network have nS (state size) inputs and nA outputs that represent the distribution over actions.

From a trajectory $\tau$ and output from the model, we obtain at each step $t$:

#### <center>$G_t(\tau) * log \pi (A_t | S_t;\Theta)$</center>

>- $G_t$ being the discounted return
>- $log \pi (A_t | S_t;\Theta)$ is parameterized by $\Theta$, so it is the output from the policy network.

So the gradient we are trying to estimate and maximize **(The Objective Function)** is:

#### <center>$\boxed{J(\Theta) = \frac{1}{T} \sum_{t=0}^T G_t(\tau) * log \pi (A_t | S_t;\Theta)}$</center>

In PyTorch, since the default behavior of gradient update is gradient descent, we put a negative sign in order to do gradient ascent.  

```python
loss = -(discounts * returns * logpas).mean()
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

As we can see, the return $G_t$ is used to **weight** the log probability of the action taken at time t. That mean if the return is bad at time t, it is because action taken at time t was bad, so by multiplying the bad return with the probability of that action, we reduce the likelihood.
of that action being selected at that step.  
  
## Vanilla Policy Gradient or REINFORCE with baseline

<center>Use 2 neural networks (policy network & value network) to respectively give the value of a state and generate probabilities over actions</center>  

  ```mermaid
%%{ init: { 'flowchart': { 'curve': 'basis' } } }%%
graph

subgraph Vanilla Policy Gradient

   subgraph Value Network
        id1((s1)) & id2((s2)) & id3((s3)) & id4((s4)) ---> 
        h1((h1)) & h2((h2)) & h3((h3)) & h4((h4)) & h5((h5)) & hn((hn)) ---> v1((v1))
    end

    subgraph Policy Network
        i1((s1)) & i2((s2)) & i3((s3)) & i4((s4)) ---> 
        hh1((h1)) & hh2((h2)) & hh3((h3)) & hh4((h4)) & hh5((h5)) & hh6((hn)) ---> a1((a1)) & a2((a2))
    end

end


```
- The Value Network have nS (state size) inputs and 1 output that represent the Value of a state.
- The Policy Network have nS (state size) inputs and nA outputs that represent the distribution over actions.

Some issues with REINFORCE:
>- **High variance** because of the accumulation of random event along a trajectory (full monte carlo return is used to calculate the gradient).
>- Log probabilities are changing proportionally to the return : $G_t(\tau) log \pi (A_t | S_t;  \Theta)$ - this can be an issue in environment with only positive rewards, return can be the same, so action probability can be quite similar.

In VPG, we solve the 2nd point: we need a way to differenciate "ok actions" & "best actions".  
For this we use the **Action-Advantage** function estimate $A(S_t, A_t; \phi) = G_t - V(S_t; \phi)$ instead of the return $G_t$ to weight the log probability of actions. 

**$A(S_t, A_t; \phi)$** center scores around 0 such that:
>- Better than average actions will have a positive value
>- Worst than average actions will have a negative value

We also use an **entropy term** $H$ weighted by $\beta$ in order to encourage exploration.  
  
So the gradient we are trying to estimate is:  

#### <center>$\boxed{A(S_t, A_t; \phi) * log \pi (A_t | S_t;  \Theta) + \beta H(\pi (S_t;\Theta))}$</center>

We can subsitute $A(S_t, A_t; \phi)$ by $G_t - V(S_t; \phi)$ and obtain:

#### <center>$\boxed{(G_t - V(S_t; \phi)) * log \pi (A_t | S_t;  \Theta) + \beta H(\pi (S_t;  \Theta))}$</center>

Loss for the Policy Network is:  

#### <center>$\boxed{L(\Theta) = -\frac{1}{N}  \sum_{n=0}^N [(G_t - V(S_t; \phi)) * log \pi (A_t | S_t;  \Theta) + \beta H(\pi (S_t;  \Theta))]}$</center>
  
In first term $G_t - V(S_t; \phi)$, the state-value function is parameterized by $\phi$. So we need a value network to return the value of a state. The second and the third term are parameterized by $ \Theta$ so we also need a policy network that return action probabilities then deduce the log and the entropy.   
   
Loss for the Value Network is:  
  
#### <center>$\boxed{L(\phi) = \frac{1}{N}  \sum_{n=0}^N [(G_t - V(S_t; \phi))^2]}$</center>
  
So in VPG we need a value network and a policy network. Because VPG is still using a full trajectory, there is no bias in the algorithm, so we assume the algorithm is "right" so cannot be considered as a "critic" (A thought by Rich Sutton. I also share his idea on that, so for me this is not an actor-critic algoritm). 

## Advantage Actor-Critic: A2C (Sharing Weight)  

<center>Use of One neural networks to update both the policy & the value network by sharing the weight</center>

```mermaid
%%{ init: { 'flowchart': { 'curve': 'basis' } } }%%
graph TD

subgraph A2C
    id1((s1)) & id2((s2)) & id3((s3)) & id4((s4)) ---> 
    h1((h1)) & h2((h2)) & h3((h3)) & h4((h4)) & h5((h5)) & hn((hn)) ---> v1((v1)) & a1((a1)) & a2((a2))
end

```


<center>Use of multiple workers (multiprocessing) to collect samples of the environment</center>  

```mermaid
%%{ init: { 'flowchart': { 'curve': 'basis' } } }%%
graph TD

subgraph Workers interacting with multiple envs
    id1((worker env 1)) & id2((worker env 2)) & id3((worker env n)) --> idenv(Multiprocessing Env class);
    idenv --> A2C_Agent
    A2C_Agent --> idenv
end
```

