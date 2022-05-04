

capdat = load('u238cap.mat');

capE = capdat.x;
capxs = capdat.y;

% find resolved resonance range - initial, total window
upper_lim_RRR = find(round(capE)==140);
lower_lim_RRR = find(round(capE)==3);
capE = capE(lower_lim_RRR:upper_lim_RRR);
capxs = capxs(lower_lim_RRR:upper_lim_RRR);
length(capE)

% capE = capE(1:1e4);
% capxs = capxs(1:1e4);
capE = capE(5.4e5:5.45e5);
capxs = capxs(5.4e5:5.45e5);

plotting = true;

if plotting
    figure(1); clf
    loglog(capE,capxs); hold on
    for i = 1:9
        xline(capE((500*i)))
    end
    xlabel('eV')
    legend('\sigma','500 energy points')
end

%% window selection
ppw = 100;
NumWindows=ceil(length(capE)/(ppw/2)) ;
EndWindow_pts = int8(length(capE)/(ppw/2)-(NumWindows-1))*(ppw/2);

WindowCrossSection=zeros(NumWindows-1,ppw);
% WindowCrossSection_std=zeros(NumWindows,ppw);
WindowEnergies=zeros(NumWindows-1,ppw) ;

Energy_start = min(capE);

for iW = 1:NumWindows-2

%     if iW == 1
%         shift = 0;
%     else
%         shift = (ppw)/2;
%     end
        
    FirstEnergyIndex = 1+(ppw/2)*(iW-1) ;
    LastEnergyIndex = FirstEnergyIndex+ppw-1;

    WindowEnergies(iW,:)= capE(FirstEnergyIndex:LastEnergyIndex);
    WindowCrossSection(iW,:)=capxs(FirstEnergyIndex:LastEnergyIndex);
%     WindowCrossSection_std(iW,:)=capxs(FirstEnergyIndex:LastEnergyIndex);

end

figure(2); clf
for iW = 1:NumWindows-1
    loglog(WindowEnergies(iW,:),WindowCrossSection(iW,:)); hold on
    
end

% for iW = 1:3
%     xline(min(WindowEnergies(iW,:))); xline(max(WindowEnergies(iW,:)))
% end

