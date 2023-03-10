%% Q3.3a disparity map algo function

function D = disparity_map(Pl, Pr, y, x)

%retrieve paameters
half_h = floor(y / 2);
half_w = floor(x / 2);
[Pl_h, Pl_w] = size(Pl);

%initialize D with matrixx of ones
D = ones(Pl_h - y + 1, Pl_w - x + 1);

%Calculate disparity
for i = half_h+1:Pl_h - half_h
    for j = half_w+1:Pl_w - half_w
        
        T = Pl(i-half_h:i+half_w,j-half_h:j+half_w);
        l = j-14;
        r = j;
        if l < half_w+1
            l = half_w+1;
        end

        min_xr = l;
        min_ssd = Inf;

        for xr = l:r
            I = Pr(i-half_h:i+half_w,xr-half_h:xr+half_w);
            ssd1 = ifft2(fft2(I) .* fft2(rot90(I, 2)));
            ssd1 = ssd1(y, x);
            ssd2 = ifft2(fft2(T) .* fft2(rot90(I, 2)));
            ssd2 = ssd2(y, x) * 2;
            ssd = ssd1 - ssd2;
            if ssd < min_ssd
                min_ssd = ssd;
                min_xr = xr;
            end
        end
        D(i, j) = j - min_xr;
    end
end
end  