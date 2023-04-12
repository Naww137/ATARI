%% linear model


% y1 = @(w) w(2)*x + w(1); 
% y2 = @(w) w(3)*x + w(2)*x + w(1); 


%%
figure(1); clf;
% for ndat = round(logspace(1,4,30))
%LRT
ndat = 200; 
nparnull = 0 ; 
nparalt = 15 ; 

dofnull = ndat-nparnull; 
dofalt = ndat-nparalt; 
likelihood_null = chi2pdf(dofnull-2, dofnull); 
likelihood_alt = chi2pdf(dofalt-2, dofalt) ;

figure(1); clf; hold on
x = linspace(0,ndat+250, 1000);
plot(x, chi2pdf(x, dofnull), "Color",'b')
yline(likelihood_null, "Color","b")
plot(x, chi2pdf(x, dofalt), "Color",'r')
yline(likelihood_alt, "Color","r")
xline(183)
    
% D = 2*( log(likelihood_alt) - log(likelihood_null) )
D = -2*(log(likelihood_null) - log(likelihood_alt))
% LD = chi2pdf(D,nparalt-nparnull) 
LD = chi2pdf(D, dofnull-dofalt) 

% semilogx(ndat, D, 'o'); hold on
% somehow constrain likelihood with SE?


% end

figure(2); clf; hold on
x = linspace(0,nparalt, 1000);
plot(x, chi2pdf(x, nparalt - nparnull), "Color",'b')
% yline(chi2pdf(D, nparalt - nparnull), "Color","b")
xline(D)