% clf



case_basename = 'slbw_2L_allexp';%"slbw_testing_1L_noexp";

% addpath("xs_functions/");
interface_directory = "/Users/noahwalton/Library/Mobile Documents/com~apple~CloudDocs/Research Projects/Resonance Fitting/sammy/";
% interface_directory = "/home/nwalton1/my_sammy/interface/";
case_dir = strcat(interface_directory,case_basename);
case_syndat_dir = strcat(interface_directory,case_basename,"/synthetic_data/");

%% import synthetic experimental data

NumPeaks = 2; % per case
plotting = true;
add_noise = false;
run_baron_bool = true;
 
true_parms = readtable(strcat(case_dir,"/true_parameters.csv"));
true_w = [true_parms{:,2:1+(NumPeaks*3)}].*repmat([1, 1e-3, 1e-3],1,NumPeaks); % widths are in meV!!!
true_w = [true_w repmat([1],length(true_w),NumPeaks)];


% solution_w = [2105.15160800000	506.740813000000*1e-3	20317.7426000000*1e-3 1];
% solution_w = [1837.37921100000	512*1e-3	9179.52334*1e-3	1967.63517300000	508.3718945*1e-3	390.4651094*1e-3    2176.58291900000	563.5903353*1e-3	11492.64564*1e-3 2667.59779800000	461.6442189*1e-3	23084.33704*1e-3	2752.96080800000	482.2181976*1e-3	197.2458461*1e-3 1 1 1 1 1];
% initial_vec = [1837.37921100000	512*1e-3	9179.52334*1e-2	1967.63517300000	508.3718945*1e-3	390.4651094*1e-2    2176.58291900000	563.5903353*1e-3	11492.64564*1e-2 2667.59779800000	461.6442189*1e-3	23084.33704*1e-2	2752.96080800000	482.2181976*1e-3	197.2458461*1e-2 1 1 1 1 1];
% initial_vec = [1837 0.1 10  1967 0.2 5   2176 0.25 20  2667 0.2 35   2752 0.2 5   1 1 1 1 1];
initial_vec = [];

% colors = repmat(["b", "r", "g"],1,3);
icolor = 0;

baron_parms = zeros(size(true_parms));

for icase = 1:1
filename = "syndat_"+string(icase);
filepath = strcat(case_syndat_dir,filename);
synth_exp_dat = readmatrix(filepath, 'FileType','text');
if size(synth_exp_dat,2) < 3
    continue
end

% Energies = synth_exp_dat(800:2500,1)' ;
% true_xs = synth_exp_dat(800:2500,2)';
% Energies = synth_exp_dat(415:2200,1)' ;
% true_xs = synth_exp_dat(415:2200,2)';
Energies = synth_exp_dat(:,1)' ;
true_xs = synth_exp_dat(:,2)';

icolor = icolor + 1 ;

if add_noise
    a=50; b=2;
    Noisy_CrossSection_std=zeros(1,length(Energies));
    Noisy_CrossSection=zeros(1,length(Energies));
    for jj=1:1
        [Noisy_CrossSection_std(jj,:),Noisy_CrossSection(jj,:)]=Noise(synth_exp_dat,a,b);
    end
end
if plotting
    figure(icase); clf
%     figure(1);
%     plot(Energies,true_xs,'.','Color', colors(icolor), 'DisplayName', strcat('Synth Data NoExp - ',string(icase))); hold on
    plot(Energies,true_xs,'.', 'DisplayName', strcat('Synth Data AllExp - ',string(icase))); hold on
%     plot(Energies,true_xs,'.','Color', colors(icolor), 'DisplayName', strcat('Synth Data AllExp - ',string(icase))); hold on
    if add_noise
        scatter(Energies,Noisy_CrossSection, 'DisplayName', 'Experimental')
        xlabel('Energy (eV)'); ylabel('\sigma')
    end
end


WC = true_xs; 
WE = Energies;
xs_func = xs_SLBW_EGgGn(NumPeaks,WE); 
Options=baronset('threads',8,'PrLevel',1,'CutOff',5,'DeltaTerm',1,'EpsA',0.1,'MaxTime',2*60);
[w, SE, fval] = run_baron(xs_func, NumPeaks, WC, WE, run_baron_bool, initial_vec, Options); 

if plotting
    solution_w = true_w(icase,:);
    plot(Energies, xs_func(solution_w), 'DisplayName','Baron Func w/TrueParm')
%     plot(Energies, xs_func(initial_vec), 'DisplayName','Baron Func w/Initial')
%     disp('SE for BaronFunc with: TrueParm, Initial, BaronSol')
%     disp(SE(solution_w));disp(SE(initial_vec))
    if run_baron_bool
%         plot(Energies, xs_func(w), 'Color', colors(icolor), 'DisplayName', strcat('Baron Sol - ',string(icase)));
        plot(Energies, xs_func(w), 'DisplayName', strcat('Baron Sol - ',string(icase)));
%         disp(SE(w))
    end
    legend()
end

if run_baron_bool
% baron_parms(icase,:) = [icase, w(1), w(2).*1e3, w(3).*1e3];

baron_parms(icase,:) = [icase, w(1), w(2).*1e3, w(3).*1e3, ...
                           w(4), w(5).*1e3, w(6).*1e3]; 

% baron_parms(icase,:) = [icase, w(1), w(2).*1e3, w(3).*1e3, ...
%                            w(4), w(5).*1e3, w(6).*1e3, ...
%                            w(7), w(8).*1e3, w(9).*1e3]; 

% baron_parms(icase,:) = [icase, w(1), w(2).*1e3, w(3).*1e3, ...
%                                w(4), w(5).*1e3, w(6).*1e3, ...
%                                w(7), w(8).*1e3, w(9).*1e3, ...
%                                w(10), w(11).*1e3, w(12).*1e3, ...
%                                w(13), w(14).*1e3, w(15).*1e3] ;
end

disp('Completed Case:')
disp(icase)
end

%%
% import baron_parms
% T = array2table(baron_parms);
% T.Properties.VariableNames(1) = "case";
% for ilevel =1:NumPeaks
%     stride = 3*(ilevel-1);
%     E="E%d"; Gg="Gg%d"; Gn="Gn%d";
%     stride_title = [sprintf(E,ilevel), sprintf(Gg,ilevel), sprintf(Gn,ilevel)];
%     T.Properties.VariableNames(2+stride:4+stride) = stride_title;
% end
% writetable(T,strcat(case_dir,'/baron_parameters.csv'))

%%
% save_name = strcat(case_basename,'.mat');
% save save_name baron_parms


%%

% ========
% function to add noise to synthetic true cross section
% ========
function [std,New_CrossSection]=Noise(CrossSection,a,b)
%Noise model to modify the cross section values that we are given from
%SAMMY, and report the std on those new cross sections values. These
%"represent" the exp cross sections that would be seen.

Trans_sigma=a*CrossSection+b;
NoisyTrans_sigma=normrnd(Trans_sigma,sqrt(Trans_sigma));
New_CrossSection=(NoisyTrans_sigma-b)/a; %
std=(sqrt(Trans_sigma))/a;
end
