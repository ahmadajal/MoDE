function d = cosine_distance(x,y)

    denom = norm(x)*norm(y);
    d = 1 - dot(x,y)/denom; 