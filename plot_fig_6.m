clear all;
data = {'spon', 'evoked', 'placebo'}
figure;hold on
for fig = 1:3
    file_name = strcat('figure_6',data{fig},'.mat');
    save_fig_name = strcat('figure_6',data{fig});
    load(file_name);
    for j = 1: 10
        for i = 1: 8
            C_uv(j,i) = corr(reshape(sum_u(1,j, :, i),[numel(sum_u(1,j, :, i)),1]), reshape(sum_v(1,j, :, i),[numel(sum_v(1,j, :, i)),1]));
        end
    end
%     figure(fig);
    errorbar(linspace(50, 400, 8), mean(C_uv,1), std(C_uv)/sqrt(10),'LineWidth', 1.);
    xlabel('Time Delay \Delta_{u}');
    ylabel('corr(A_{u},A_{v})');
    title('Correlations of Pain Conditions with Changing \Delta_{u}');
    grid;
%     if fig == 1
%         title('Evoked Pain Correlation with Changing \Delta_{u}');
%     elseif fig == 2
%         title('Spontaneous Pain Correlation with Changing \Delta_{u}');
%     else
%         title('Placebo Pain Correlation with Changing \Delta_{u}');
%     end
    
end
legend({'Spontaneous', 'Evoked', 'Placebo'}, 'Location', 'best');
