function y = proj_l_u(x,l,u)
%% Project on [l,u]; works with vectors as well
%l: lower bound
%u: upper bound
%x: input

y = min(max(x,l),u);