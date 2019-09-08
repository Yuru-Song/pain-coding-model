clear all;

fig_path = '/figures/';
data_path = '/data/';

data = {'a','b','c','d','e'};
for fig = 1: 5
    file_name = strcat(data_path,'/figure_5',data{fig},'.mat');
    save_fig_name = strcat(fig_path,'/figure_5',data{fig});
    load(file_name);
    for j = 1: 10
        for i = 1: 50
            C_uv(j,i) = corr(reshape(sum_u(1,j, :, i),[numel(sum_u(1,j, :, i)),1]), reshape(sum_v(1,j, :, i),[numel(sum_v(1,j, :, i)),1]));
        end
    end
    figure(fig);
    errorbar(linspace(0, 10, 50),mean(C_uv,1), std(C_uv)/sqrt(10));
%     title(data{fig}); 
    if mod(fig,2) == 1
        xlabel('\Pi_2');
    else
        xlabel('\Pi_1, \Pi_3');
    end
    ylabel('corr(A_u,A_v)');
    set(gcf, 'Units','Normalized','OuterPosition',[(fig-1)*.22 + .1 ,.6,.21,.35]);
    saveas(fig, save_fig_name,'epsc');
    saveas(fig, save_fig_name,'fig');
end
% pause;close all;