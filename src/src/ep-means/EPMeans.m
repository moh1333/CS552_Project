%Michael Harding
%CS 542
%Project Work
%
%EP-Means algorithm for clustering word densities

%Parameters
K = 4;    %num centroids


tempMaps = readNPY('densityMaps_new.npy');
numwords = size(tempMaps,1);

%reshape maps
maps = zeros(100,232,numwords);
for i = 1:numwords
    for j = 1:100
        for k = 1:232
            maps(j,k,i) = tempMaps(i,j,k);
        end
    end
end

%regularize the maps
maps = exp(maps);
for i = 1:numwords
    maps(:,:,i) = maps(:,:,i)/sum(sum(maps(:,:,i)));
end

%convert maps into 1-dim CDFs
%and reduce dimensionality
cdfs = zeros(numwords, 23200);
dim = 0;  %new dimensionality of CDFs
for i = 1:100
    for j = 1:232
        if(maps(i,j,1) ~= 0)
            if(dim == 0)
                dim = dim + 1;
                cdfs(:, dim) = reshape(maps(i, j, :),[numwords,1]);
            else
                dim = dim + 1;
                cdfs(:, dim) = cdfs(:, dim-1) + reshape(maps(i, j, :),[numwords,1]);
            end
        end
    end
end
cdfs = cdfs(:,1:dim);

centroids = init(cdfs,K,numwords);
[newcents,~] = kmeans(cdfs, centroids, K, numwords);
counter = 1;
while(sum(EMD(centroids',newcents').^2) > 10e-5)
    centroids = newcents;
    [newcents, ~] = kmeans(cdfs, centroids, K, numwords);
    disp(counter);
    counter = counter + 1;
end
[~, clusters] = kmeans(cdfs, centroids, K, numwords);

for i = 1:K
    plot(centroids(i,:));
    hold on
end
hold off


function [new_cents, mins_inds] = kmeans(data, centroids, k, numwords)
dist = zeros(k,numwords);
for i = 1:k
    dist(i,:) = EMD(data',centroids(i,:)');
end
mins_inds = zeros(numwords,2);
mins_inds(:,2) = [1:numwords]';
[~,labs] = min(dist.^2);
mins_inds(:,1) = labs';
mins_inds = sortrows(mins_inds);

new_cents = zeros(size(centroids));
m = 1;
for i = 1:k
    n = m;
    while((m <= numwords) && (mins_inds(m,1) == i))
        m = m+1;
    end
    cluster = data(mins_inds(n:(m-1),2),:);
    new_cents(i,:) = sum(cluster)/size(cluster,1);
end
end

function centroids = init(data, k, numwords)
centroids = zeros(k, size(data,2));
centroids(1,:) = data(randi(numwords),:);
dist = zeros(k,numwords);
for i = 1:(k - 1)
    dist(i,:) = EMD(data',centroids(i,:)');
    if(i > 1)
        D2 = min(dist(1:i,:).^2);
    else
        D2 = dist(1,:).^2;
    end
    D2 = D2/sum(D2);
    index = datasample(1:numwords, 1, 'Weights', D2);
    centroids(i+1,:) = data(index,:);
end
end

function dist = EMD(x,y)
dist = sum(abs(x - y));
end