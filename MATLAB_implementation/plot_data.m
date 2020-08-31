function plot_data(X_2d, Score, method, Rd, Rorder, Rc, data)
%% This function generates the 2D scatter plot of the embedded data points
    [~,ind_temp] = sort(Score,'ascend');
    [~,ind] = sort(ind_temp); % Sort score, the higher the score correponding to higher color in colorbar
    N = size(X_2d, 1);
    c=colormap(parula(N));

    figure;
    scatter(X_2d(:,1),X_2d(:,2),25, c(ind,:),'filled')
    set(gca, 'XTick', []);
    set(gca, 'YTick', []);

    box on;
    colorbar('Ticks',[0.2,0.9],...
             'TickLabels',{'Low Score','High Score'});

    new_title = sprintf('%s: R_d=%0.3f, R_o=%0.3f, R_c=%0.3f', method, Rd, Rorder, Rc)
    title(new_title, 'fontsize', 15);
   % fname = fprintf('figures/%s.png', method);
    data_name = split(data, ".");
    saveas(gcf,join(['figures/', method, '_', data_name(1)], ""), 'jpeg');
end
