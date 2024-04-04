---
layout: default
---

<style>
#search-input {
  padding: 10px;
  margin-bottom: 30px;
  border: 1px solid #ccc;
  border-radius: 8px;
  font-size: 16px;
  outline: none;
}
</style>

<div id="search-container">
  <input type="text" id="search-input" name="q" value="" autofocus>
  <ul id="results-container" class="post-list" ></ul>
</div>

<script src="{{site.baseurl}}/js/search.js" type="text/javascript"></script>

<script>
SimpleJekyllSearch({
  searchInput: document.getElementById('search-input'),
  resultsContainer: document.getElementById('results-container'),
  json: '{{site.baseurl}}/search.json',
  sortMiddleware: function(a, b) {
      aPrio = matchPriority(a.matchedField);
      bPrio = matchPriority(b.matchedField);
      return bPrio - aPrio;
    },
  searchResultTemplate: "<li class=''>" +
    "<div class='content'>" +
      "<span class='post-meta'>{date}</span>" +
      "<a class='post-link' href='/blog{url}'>{title}</a>" +
      "<p>{excerpt}</p>" +
    "</div>" +
  "</li>",  })

  function matchPriority(fieldMatched) {
    switch (fieldMatched) {
      case 'tags':
        return 5;
      case 'title':
        return 4;
      case 'excerpt':
        return 3;
      default:
        return 0;
    }
  }

  window.addEventListener('load', function() {
      var searchParam = new URLSearchParams(window.location.search).get("q");
      if (searchParam != null) {
          document.getElementById('search-input').value = searchParam;
          setTimeout(() => {
            sjs.search(searchParam);
          }, 100);
      }
      document.getElementById('search-input').placeholder = "Type your search here...";}, false);
</script>
