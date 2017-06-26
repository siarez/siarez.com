$(function() {
    $('img.zoomable').on('click', function() {
        $('.enlargeImageModalSource').attr('src', $(this).attr('src'));
        $('#enlargeImageModal').modal('show');
    });
});

/*
$( document ).ready(function() {
    new Tether({
        element: '.stickyNav',
        target: '.postcontent',
        attachment: 'top left',
        targetAttachment: 'top right',
        constraints: [{
            to: 'window',
            pin: true
        }]
    });
});
*/
