function table_out = read_hdf5(case_file, dataset)

float_data = h5read(case_file, sprintf('%s/block0_values', dataset));
label_data = h5read(case_file, sprintf('%s/block0_items',dataset));

%remove whitespace
for i = 1:length(label_data)
    label_data(i) = deblank(label_data(i));
end

table_out = array2table(float_data','VariableNames',cellstr(label_data'));

end
