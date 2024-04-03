---
layout: default
---

<style>
#search-input {
  padding: 10px;
  border: 1px solid #ccc;
  border-radius: 5px;
  font-size: 16px;
}
</style>

<div id="search-container">
  <input type="text" id="search-input" placeholder="search...">
  <ul id="results-container"></ul>
</div>

<script src="{{site.baseurl}}/js/search.js" type="text/javascript"></script>

<script>
SimpleJekyllSearch({
  searchInput: document.getElementById('search-input'),
  resultsContainer: document.getElementById('results-container'),
  json: '{{site.baseurl}}/search.json',
  searchResultTemplate: "<li>" +
          "<h3>" +
            "<a href='{url}'>{title}</a>" +
          "</h3>" +
        "</li>"
})
</script>
