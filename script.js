let originalTriplets;
let leftoverTriplets;
let inBounds;
let root = 'webpage_assets/';
let tripletDict = {
    'triplet1': {'Humans': [0.0, 1.0], 
                    'LPIPS': [0.5848354697227478, 0.637165367603302], 
                    'DINO': [0.2728973627090454, 0.2907627820968628], 
                    'CLIP': [0.047773540019989014, 0.05663299560546875], 
                    'Ours': [0.20159006118774414, 0.17979753017425537]},
    'triplet2': {'Humans': [1.0, 0.0], 
                    'LPIPS': [0.7032673954963684, 0.6846778392791748], 
                    'DINO': [0.1351228952407837, 0.18742769956588745], 
                    'CLIP': [0.053246378898620605, 0.04778444766998291], 
                    'Ours': [0.194116473197937, 0.2475385069847107]},
    'triplet3': {'Humans': [0.0, 1.0], 
                    'LPIPS': [0.6661412119865417, 0.767113208770752], 
                    'DINO': [0.21639931201934814, 0.31087344884872437], 
                    'CLIP': [0.06859838962554932, 0.09371405839920044], 
                    'Ours': [0.3725281357765198, 0.2923005223274231]},
    'triplet4': {'Humans': [0.0, 1.0], 
                    'LPIPS': [0.6822073459625244, 0.6929465532302856], 
                    'DINO': [0.3839675188064575, 0.4142078161239624], 
                    'CLIP': [0.1335831880569458, 0.13970071077346802], 
                    'Ours': [0.5118826627731323, 0.41823631525039673]},
    'triplet5': {'Humans': [1.0, 0.0], 
                    'LPIPS': [0.7918524742126465, 0.7587180733680725], 
                    'DINO': [0.19339030981063843, 0.183249831199646], 
                    'CLIP': [0.09237241744995117, 0.10501706600189209], 
                    'Ours': [0.20740431547164917, 0.23512232303619385]},
    'triplet6': {'Humans': [0.0, 1.0], 
                    'LPIPS': [0.793615996837616, 0.8075497150421143], 
                    'DINO': [0.2667297124862671, 0.23749792575836182], 
                    'CLIP': [0.07091611623764038, 0.10216999053955078], 
                    'Ours': [0.33677661418914795, 0.29327940940856934]},
    'triplet7': {'Humans': [0.0, 1.0],
                    'LPIPS': [0.8293099999427795, 0.8551262021064758],
                    'DINO': [0.2980373501777649, 0.37643980979919434],
                    'CLIP': [0.12851327657699585, 0.21276861429214478],
                    'Ours': [0.3657013177871704, 0.23976093530654907]},
};

function init() {
    setTripletTracker();
    setDragListener();
    $("a").attr("target", "_blank");
    document.getElementById('bibtex_link').target="_self"
}

function setTripletTracker() {
    originalTriplets = [];
    for (let i =1; i <= 7; i++) {
        originalTriplets.push($('#draggable-triplet-' + i).get(0));
    }
    leftoverTriplets = originalTriplets.map((x) => x);
}

function setDragListener() {
    // let bigContainer = $('#big-container').get(0);
    // let bigBounds = bigContainer.getBoundingClientRect();
    // let startX = bigBounds.x;
    // let startY = bigBounds.y;

    // let tripletWidth = $('.draggable-triplet').width();
    // let tripletHeight = $('.draggable-triplet').height();

    // let endX = startX + bigBounds.width - tripletWidth;
    // let endY = startY + bigBounds.height - tripletHeight;

    $('.draggable-triplet')
    .mouseover(function() {
        $(this).css('opacity', '0.7');
    })
    .mouseout(function() {
        $(this).css('opacity', '1.0');
    })
    .draggable({
        containment: $('#big-container').get(0),
        // containment: [$('#big-container').get(0).getBoundingClientRect().x + $(window).scrollLeft(),
        // $('#big-container').get(0).getBoundingClientRect().y + $(window).scrollTop(),
        // $('#big-container').get(0).getBoundingClientRect().x + $(window).scrollLeft() + $('#big-container').get(0).getBoundingClientRect().width - $('.draggable-triplet').width(),
        // $('#big-container').get(0).getBoundingClientRect().y + $(window).scrollTop() + $('#big-container').get(0).getBoundingClientRect().height - $('.draggable-triplet').height()],
        drag: function(event, ui) {
            tripletWidth = $('.draggable-triplet').width();
            tripletHeight = $('.draggable-triplet').height();
            
            let tripletLeft = ui.helper.position().left;
            let tripletTop = ui.helper.position().top;
            let tripletRight = tripletLeft + tripletWidth;
            let tripletBottom = tripletTop + tripletHeight;

            let dropBounds = $('#drop-container').get(0).getBoundingClientRect();
            let startXDrop = dropBounds.left + $(window).scrollLeft();
            let startYDrop = dropBounds.top + $(window).scrollTop();

            let endXDrop = dropBounds.right + $(window).scrollLeft();
            let endYDrop = dropBounds.bottom + $(window).scrollTop();

            let specificTriplet = $('#' + $(this).attr('id'));
            let idx = leftoverTriplets.indexOf(specificTriplet);
            if (idx != -1) {
                leftoverTriplets.splice(idx);
            }
            
            console.log('startXDrop', 'startYDrop', 'endXDrop', 'endYDrop');
            console.log(startXDrop, startYDrop, endXDrop, endYDrop);
            console.log('tripletLeft', 'tripletTop', 'tripletRight', 'tripletBottom');
            console.log(tripletLeft, tripletTop, tripletRight, tripletBottom);
            console.log('(tripletRight >= startXDrop && tripletLeft <= endXDrop) && (tripletBottom >= startYDrop && tripletTop <= endYDrop)');
            console.log('\n');

            if ((tripletRight >= startXDrop && tripletLeft <= endXDrop) &&
                (tripletBottom >= startYDrop && tripletTop <= endYDrop)) {
                $('#drop-container').css('background-color', '#eee');
                snapToDropContainer(ui, $(this), true);
                inBounds = true;
            } else {
                $('#drop-container').css('background-color', '#fff');
                snapToDropContainer(ui, $(this), false);
                inBounds = false;
            }
        },
        stop: function(event, ui) {
            let triplet = $(this);
            if (inBounds) {
                setMetrics(triplet);
                $('#metrics-1').css('visibility', 'visible');
                $('#metrics-3').css('visibility', 'visible');
            }
            triplet.css('top', 0);
            triplet.css('left', 0);
            leftoverTriplets = originalTriplets.map((x) => x);
            $('#drop-container').css('background-color', '#fff');
        },
        start: function(event, ui) {
            for (let i in leftoverTriplets) {
                let triplet = $(leftoverTriplets[i]);
                triplet.css('visibility', 'visible');
            }
            $('.metrics').css('visibility', 'hidden');
        }
    });
}

function setMetrics(triplet) {
    let dir = root + 'metric_icons/';
    let idx = triplet.attr('id');
    idx = idx[18];
    let leftContainer = $('#metrics-1');
    let midContainer = $('#metrics-2');
    let rightContainer = $('#metrics-3');

    let leftMetricContainer;
    let midMetricContainer;
    let rightMetricContainer;

    let metricLeft;
    let metricRight;
    let leftMetric;


    let metrics = ['LPIPS', 'DINO', 'CLIP', 'Ours', 'Humans'];

    for (let i in metrics) {
        leftMetricContainer = leftContainer.children().eq(i).children().eq(0);
        midMetricContainer = midContainer.children().eq(i).children().eq(0);
        rightMetricContainer = rightContainer.children().eq(i).children().eq(0);

        let metric = metrics[i];
        metricLeft = tripletDict['triplet' + idx][metric][0];
        metricRight = tripletDict['triplet' + idx][metric][1];

        if (metric === 'Humans') {
            leftMetric = metricLeft > metricRight;
        } else {
            leftMetric = metricLeft < metricRight;
        }
    
        leftMetricContainer.attr('src', dir + metric.toLowerCase() + '_' + (leftMetric + 0) + '.svg');
        rightMetricContainer.attr('src', dir + metric.toLowerCase() + '_' + (1 - (leftMetric + 0)) + '.svg');
        midMetricContainer.attr('src', dir + metric.toLowerCase() + '_0.svg');

        leftValContainer = leftContainer.children().eq(i).children().eq(1);
        midValContainer = midContainer.children().eq(i).children().eq(1);
        rightValContainer = rightContainer.children().eq(i).children().eq(1);

        if (metric === 'Humans') {
            leftValContainer.html('  ');
            midValContainer.html('  ');
            rightValContainer.html('  ');
        } else {
            leftValContainer.html(metricLeft.toFixed(2).toString().substring(1));
            midValContainer.html(Number(.01).toString().substring(1));
            rightValContainer.html(metricRight.toFixed(2).toString().substring(1));
        }

        if (leftMetric) {
            leftValContainer.css('font-weight', '700');
            rightValContainer.css('font-weight', '400');
        } else {
            rightValContainer.css('font-weight', '700');
            leftValContainer.css('font-weight', '400');
        }
    }
}

function snapToDropContainer(ui, triplet, snap) {
    if (snap) {
        path1 = triplet.children().eq(0).attr('src');
        path2 = triplet.children().eq(1).attr('src');
        path3 = triplet.children().eq(2).attr('src');
        triplet.css('visibility', 'hidden');
    } else {
        path1 = '';
        path2 = '';
        path3 = '';
        triplet.css('visibility', 'visible');
    }

    $('#active-img-1').attr('src', path1);
    $('#active-img-2').attr('src', path2);
    $('#active-img-3').attr('src', path3);
}

//$(document).ready(init);