function [cHandle] = scatterXY(X, Y, numBins, onOff)
[n, c] = hist3([X' Y'], [numBins numBins]);
d = n./max(max(n));
e = interp2([c{1}], [c{2}], d', X, Y);
[~, b] = sort(e, 'descend');
scatter(X(b), Y(b), 20, e(b), 'filled'); 
if (onOff)
    cHandle = colorbar;
    ylabel(cHandle, 'Relative Density of Data Points', 'FontSize', 12, ...
        'FontName', 'Arial', 'FontWeight', 'Bold');
    caxis([0 1]);
    set(cHandle, 'YTick', [])
    colormap parula
end
