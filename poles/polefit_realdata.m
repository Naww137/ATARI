
cap_dat = load('U238cap.mat');

figure(1); clf
loglog(cap_dat.x,cap_dat.y); title('Total Reconstructed \sigma'); 

% find resolve resonance range and less-dense energy grid
RRR_index = find(cap_dat.x>3.5 & cap_dat.x<140);
course_RRR_index = RRR_index(1:5:end);

capE = cap_dat.x(course_RRR_index); capXS = cap_dat.y(course_RRR_index);

figure(2); clf
loglog(capE,capXS); title('Course Reconstructed \sigma in RRR'); 
