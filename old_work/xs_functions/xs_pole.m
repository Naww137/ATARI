function xs = xs_pole(NumPeaks,Energies)
    z = Energies;
    xs = @(w) 0; 
    for iRes = 1:NumPeaks
    %     f = @(rr,irip,rp,ipsqr) -irip(iRes)./((rp(iRes)-z).^2+ipsqr) + rr(iRes).*(rp(iRes)-z)./((rp(iRes)-z).^2+ipsqr(iRes)) ;
        xs = @(w) xs(w) + -w(2+4*(iRes-1))./((w(3+4*(iRes-1))-z).^2+w(4+4*(iRes-1))) + w(1+4*(iRes-1)).*(w(3+4*(iRes-1))-z)./((w(3+4*(iRes-1))-z).^2+w(4+4*(iRes-1))) ;
    end
end