% primary controls

plot_random_windows = false;








%% load in data
cap_dat = load('U238cap.mat');

% figure(1); clf
% loglog(cap_dat.x,cap_dat.y); title('Total Reconstructed \sigma'); 

% find resolve resonance range and less-dense energy grid
RRR_index = find(cap_dat.x>3.5 & cap_dat.x<140);
course_RRR_index = RRR_index(1:2:end);

capE = cap_dat.x(course_RRR_index); capXS = cap_dat.y(course_RRR_index);

% figure(2); clf
% loglog(capE,capXS); title('Course Reconstructed \sigma in RRR'); 

%% break the data up into tractable windows of ~500 energy points

ppw = 500;
FullWindows = floor(length(capE)/(ppw/2));
Ewindows = zeros(FullWindows-1, ppw); XSwindows = zeros(FullWindows-1, ppw);

for iW = 1:FullWindows-1
    FirstEnergyIndex = 1+(ppw/2)*(iW-1) ;
    LastEnergyIndex = FirstEnergyIndex+ppw-1;
    Ewindows(iW,:) = capE(FirstEnergyIndex:LastEnergyIndex);
    XSwindows(iW,:) = capXS(FirstEnergyIndex:LastEnergyIndex);
end

LastEnergyIndex_LastFullWindow = 1+(ppw/2)*(iW-1) + 500-1;
LeftoverEnergyPoints = length(capE) - LastEnergyIndex_LastFullWindow;


%% plot random windows
if plot_random_windows
    show_grid_spacing = false ;
    number_of_sample_cases = 1 ;

    for iW = randi(818,1,number_of_sample_cases)
        if show_grid_spacing
            figure(iW); clf
            plot(Ewindows(iW,:), XSwindows(iW,:));hold on
            ylabel('Cross Section')
        %     figure(iW+1);clf
            yyaxis right
            plot(Ewindows(iW,1:end-1),diff(Ewindows(iW,:)))
            ylabel('Spacing in energy grid')
            xlabel('eV')
        else
            figure(iW); clf
            loglog(Ewindows(iW,:), XSwindows(iW,:));hold on
            ylabel('Cross Section')
        end
    end
end

%% solve each window


