function [q_val,V] = failed_update(q_val,V,reinf,predicted_value,pre_state,pre_action,cur_state,x_hat)
%FAILED_UPDATE Summary of this function goes here
%   Detailed explanation goes here
global ALPHA BETA GAMMA LAMBDA;
    q_val(pre_state,pre_action)= q_val(pre_state,pre_action)+ ALPHA*(reinf + GAMMA*V(cur_state) - V(pre_state)  + GAMMA*predicted_value - q_val(pre_state,pre_action));
    
    for i=1:162
    V(i) = V(i) + BETA*(reinf + GAMMA* V(cur_state) - V(pre_state)) * x_hat(cur_state) ; 
    end
    
end

