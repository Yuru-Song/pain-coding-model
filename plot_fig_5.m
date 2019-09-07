clear all;
data = {'a','b','c','d','e','f'};
for fig = 1: 6
    file_name = strcat('figure_5',data{fig},'.mat');
    save_fig_name = strcat('figure_5',data{fig});
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
     ylabel('corr(A_u,A_v)')
    if fig == 1
        title('Spontaneous Pain Correlation Bet. A_{u} and A_{v} w. constant \pi_{1} and \pi_{3}')
    elseif fig == 2
        title('Spontaneous Pain Correlation Bet. A_{u} and A_{v}  w. constant \pi_{2}');
    elseif fig == 3
        title('Placebo Pain Correlation Bet. A_{u} and A_{v} w. constant \pi_{1} and \pi_{3}');
    elseif fig == 4
        title('Placebo Pain Correlation Bet. A_{u} and A_{v} w. constant \pi_{2}');
    elseif fig ==5
        title('Evoked Pain Correlation Bet. A_{u} and A_{v} w. constant \pi_{1} and \pi_{3}');
    elseif fig ==6
        title('Evoked Pain Correlation Bet. A_{u} and A_{v} w. constant \pi_{2}');
    end
    set(gcf, 'Units','Normalized','OuterPosition',[(fig-1)*.22 + .1 ,.6,.21,.35]);
    saveas(fig, save_fig_name,'epsc');
    saveas(fig, save_fig_name,'fig');
end
