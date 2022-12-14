function xs = xs_SLBW(energies, E_levels, Gg, gn, pig, P, k)

xs = zeros(1,length(energies));
for jj = 1:length(E_levels)
    xs = xs + (pig./k(energies).^2).*((2*P(energies)*gn(jj)^2*Gg(jj)) ...
        ./((energies-E_levels(jj)).^2+((2*P(energies)*gn(jj)^2+Gg(jj))/2).^2)) ;
end

end

% total (non-baron format)

% for jj = 1:1
% 
% gns = Gn(jj)./2./P(Elevels(jj)) ;
% Gn = gns.*2.*P(WE); 
% Gt = Gc(jj)+Gn(jj) ; 
% d = (Elevels(jj)-WE).^2 + (Gt./2).^2 ; 
% test = test + (2.*pig./kE.^2) .* ( (1- (1-(Gt*Gn./2./d)) ...
%                                             .*cos(2.*phi) ...
%                                          - (Elevels(jj)-WE).*Gn./d ...
%                                             .*sin(2.*phi) ) )    ;  
% end