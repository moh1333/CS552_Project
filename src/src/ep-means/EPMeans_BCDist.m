%Michael Harding
%CS 542
%Project Work
%
%EP-Means algorithm for clustering word densities
%modified to use Bhattacharyya distance with 2 dim
%probability distributions

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

%compute the k-means
centroids = init(maps,K);
[newcents,~] = kmeans(maps, centroids, K);
num_its = 1;
while(sum(DB(centroids,newcents).^2) > 10e-5)
    centroids = newcents;
    [newcents, ~] = kmeans(maps, centroids, K);
    num_its = num_its + 1;
end
[~, clusters] = kmeans(maps, centroids, K);

diff_from_noise = DB(centroids,maps(:,:,1));




function [new_cents, mins_inds] = kmeans(data, centroids, k)
dist = zeros(k,size(data,3));
for i = 1:k
    dist(i,:) = DB(data,centroids(:,:,i));
end
mins_inds = zeros(size(data,3),2);
mins_inds(:,2) = [1:size(data,3)]';
[~,labs] = min(dist.^2);
mins_inds(:,1) = labs';
mins_inds = sortrows(mins_inds);

new_cents = zeros(size(centroids));
m = 1;
for i = 1:k
    n = m;
    while((m <= size(data,3)) && (mins_inds(m,1) == i))
        m = m+1;
    end
    cluster = data(:,:,mins_inds(n:(m-1),2));
    new_cents(:,:,i) = sum(cluster,3)/size(cluster,3);
end
end

function centroids = init(data, k)
centroids = zeros(size(data,1), size(data,2),k);
centroids(:,:,1) = data(:,:,randi(size(data,3)));
dist = zeros(k,size(data,3));
for i = 1:(k - 1)
    dist(i,:) = DB(data,centroids(:,:,i));
    if(i > 1)
        D2 = min(dist(1:i,:).^2);
    else
        D2 = dist(1,:).^2;
    end
    D2 = D2/sum(D2);
    index = datasample(1:size(data,3), 1, 'Weights', D2);
    centroids(:,:,i+1) = data(:,:,index);
end
end

function dist = DB(x,y)
dist = reshape(-log(sum(sum(sqrt(x.*y)))),1,[]);
end