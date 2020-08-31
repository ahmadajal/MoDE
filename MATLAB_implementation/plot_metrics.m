function plot_metrics(Rc, Rd, Rd_full, Rs, R_order, method_name)
%% print the computed metrics for the embedding method
fprintf('\n------ Results of %s ------\n', method_name);
fprintf('---------------------------\n');
fprintf('Rc: %0.3f\n', Rc);
fprintf('Rd: %0.3f\n', Rd);
fprintf('Rd_full: %0.3f\n', Rd_full);
fprintf('Rs: %0.3f\n', Rs);
fprintf('Rorder: %0.3f\n', R_order);
fprintf('---------------------------\n');

end
